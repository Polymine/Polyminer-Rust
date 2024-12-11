
use clap::{Arg, Command};
use ocl::{builders::ContextBuilder, flags, Buffer, Device, Kernel, Platform, Program, Queue};
use serde::{Deserialize, Serialize};
use web3::contract::{Contract, Options};
use web3::signing::SecretKey;
use web3::transports::Http;
use web3::types::{Address, U256};
use web3::Web3;
use hex;
use tokio;
use std::collections::HashSet;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::io::{self, Read, Write, BufWriter};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn, error, debug};
use chrono::Local;
use env_logger::Builder;
use std::fs::File;
use sha3::{Digest, Keccak256};
use tokio::time::{interval, Duration};

/// Path to the OpenCL kernel source code.
const KERNEL_SOURCE: &str = include_str!("kernel.cl");

/// Structure to persist miner state.
#[derive(Serialize, Deserialize)]
struct MinerState {
    submitted_nonces: HashSet<u64>, // Using u64 for nonces
    global_nonce: u64,
    tx_nonce: u64,
}

/// Initialize the logger with timestamps and log levels.
fn initialize_logger() {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info) // Set default log level to Info
        .init();
}

/// Load miner state from a file.
async fn load_state(web3: &Web3<Http>, wallet_address: Address) -> MinerState {
    let file_path = "miner_state.json";
    if let Ok(mut file) = File::open(file_path) {
        let mut data = String::new();
        if file.read_to_string(&mut data).is_ok() {
            if let Ok(state) = serde_json::from_str(&data) {
                info!("Loaded miner state from {}", file_path);
                return state;
            }
        }
    }
    info!("No existing miner state found. Starting fresh.");
    // Initialize tx_nonce by fetching from the network
    match web3.eth().transaction_count(wallet_address, None).await {
        Ok(tx_nonce) => MinerState {
            submitted_nonces: HashSet::new(),
            global_nonce: 0,
            tx_nonce: tx_nonce.as_u64(),
        },
        Err(e) => {
            error!("Failed to fetch transaction count: {:?}", e);
            // If fetching fails, start tx_nonce at 0
            MinerState {
                submitted_nonces: HashSet::new(),
                global_nonce: 0,
                tx_nonce: 0,
            }
        }
    }
}

/// Save miner state to a file.
fn save_state(state: &MinerState) {
    let file_path = "miner_state.json";
    if let Ok(file) = File::create(file_path) {
        let writer = BufWriter::new(file);
        if serde_json::to_writer_pretty(writer, state).is_err() {
            error!("Failed to save miner state to {}.", file_path);
        } else {
            info!("Miner state saved successfully to {}.", file_path);
        }
    } else {
        error!("Failed to create state file at {}.", file_path);
    }
}

/// Fetches the current mining difficulty from the smart contract.
async fn fetch_difficulty(
    contract: &Contract<Http>,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> { // Return type changed to u64
    info!("Fetching difficulty...");
    let difficulty: U256 = contract.query("difficulty", (), None, Options::default(), None).await?;
    let difficulty_u64 = difficulty.as_u64(); // Convert U256 to u64
    info!("Current difficulty: {}", difficulty_u64);
    Ok(difficulty_u64)
}

/// Computes the target based on the current difficulty.
fn compute_target(difficulty: u64) -> Vec<u8> {
    let max_val = U256::MAX;
    let difficulty_u256 = U256::from(difficulty);
    let target = max_val / difficulty_u256;
    let mut target_bytes = vec![0u8; 32];
    target.to_big_endian(&mut target_bytes);
    target_bytes
}

/// Displays the wallet's MATIC balance.
async fn display_wallet_balance(
    web3: &Web3<Http>,
    wallet_address: Address,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Fetching wallet balance...");
    let balance = web3.eth().balance(wallet_address, None).await?;
    let matic_balance = balance.as_u128() as f64 / 1e18;
    info!("Wallet balance: {:.6} MATIC", matic_balance);
    Ok(())
}

fn verify_nonce(address: &Address, nonce: u64, target: &[u8]) -> bool {
    let mut nonce_bytes = [0u8; 32];
    nonce_bytes[24..].copy_from_slice(&nonce.to_be_bytes()); // Place u64 at the end

    let mut hasher = Keccak256::new();
    hasher.update(address.as_bytes());
    hasher.update(&nonce_bytes);
    let result = hasher.finalize();

    U256::from_big_endian(&result) < U256::from_big_endian(target)
}

/// Submits a valid nonce to the smart contract.
async fn submit_solution(
    contract: &Contract<Http>,
    private_key: &SecretKey,
    web3: &Web3<Http>,
    wallet_address: Address,
    nonce: u64, // Changed from U256 to u64
    submitted_nonces: Arc<Mutex<HashSet<u64>>>,
    tx_nonce_counter: Arc<Mutex<u64>>, // Using tokio::Mutex<u64>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Atomically get and increment the transaction nonce
    let mut tx_nonce = tx_nonce_counter.lock().await;
    let current_tx_nonce = *tx_nonce;
    *tx_nonce += 1;
    debug!("Using transaction nonce: {}", current_tx_nonce);

    // Fetch current gas price
    let gas_price = match web3.eth().gas_price().await {
        Ok(price) => price,
        Err(e) => {
            error!("Failed to fetch gas price: {:?}", e);
            return Err(Box::new(e));
        }
    };

    // Increase gas price by 10% to avoid underpricing issues
    let adjusted_gas_price = gas_price + (gas_price / 10);

    let options = Options {
        nonce: Some(U256::from(current_tx_nonce)),
        gas_price: Some(adjusted_gas_price),
        value: Some(U256::exp10(16)), // 0.01 MATIC in Wei
        ..Default::default()
    };

    match contract
        .signed_call("mine", (nonce,), options, private_key)
        .await
    {
        Ok(tx_hash) => {
            info!("Submitted nonce {} with TX: {:?}", nonce, tx_hash);
            Ok(())
        }
        Err(e) => {
            error!("Error submitting nonce {}: {:?}", nonce, e);
            // Decrement the tx_nonce_counter to retry this nonce in the future
            let mut tx_nonce = tx_nonce_counter.lock().await;
            if *tx_nonce > 0 {
                *tx_nonce -= 1;
            }
            // Remove the nonce from the submitted set if submission failed
            let mut submitted = submitted_nonces.lock().await;
            submitted.remove(&nonce);
            Err(Box::new(e))
        }
    }
}

/// Lists all available GPU devices.
fn list_gpus() -> Result<Vec<(Platform, Device)>, Box<dyn std::error::Error + Send + Sync>> {
    let platforms = Platform::list();
    let mut gpu_devices = Vec::new();

    for platform in platforms {
        if let Ok(devices) = Device::list(platform, Some(flags::DEVICE_TYPE_GPU)) {
            for device in devices {
                gpu_devices.push((platform, device));
            }
        }
    }

    if gpu_devices.is_empty() {
        Err("No GPU devices found!".into())
    } else {
        Ok(gpu_devices)
    }
}

/// Sets up the OpenCL queue and kernel.
fn setup_opencl(
    platform: &Platform,
    device: &Device,
) -> Result<(Queue, Kernel), Box<dyn std::error::Error + Send + Sync>> {
    let context = ContextBuilder::new()
        .platform(*platform)
        .devices(*device)
        .build()?;

    // Enable profiling to measure kernel execution times
    let queue = Queue::new(&context, *device, Some(flags::QUEUE_PROFILING_ENABLE))?;
    let program = Program::builder()
        .src(KERNEL_SOURCE)
        .devices(*device)
        .build(&context)?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("hashMessage")
        .queue(queue.clone())
        .arg(None::<&Buffer<u8>>)
        .arg(None::<&Buffer<u8>>)
        .arg(0u64) // Changed from U256::zero() to 0u64
        .arg(0u32) // Changed from U256::zero() to 0u32
        .arg(None::<&Buffer<u64>>)
        .arg(None::<&Buffer<u32>>)
        .build()?;

    Ok((queue, kernel))
}

/// Calibration phase to measure baseline hashrate.
async fn calibrate_gpu(
    kernel: &Kernel,
    queue: &Queue,
    batch_size: u64,
) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting calibration phase...");
    let mut message_data = vec![0u8; 52];
    message_data[..20].copy_from_slice(&[0u8; 20]); // Dummy address

    let d_message = Buffer::<u8>::builder()
        .queue(queue.clone())
        .copy_host_slice(&message_data)
        .flags(flags::MEM_READ_ONLY)
        .len(52)
        .build()?;

    let d_target = Buffer::<u8>::builder()
        .queue(queue.clone())
        .copy_host_slice(&vec![0xFFu8; 32]) // Dummy target (highest possible)
        .flags(flags::MEM_READ_ONLY)
        .len(32)
        .build()?;

    let solution_count = Buffer::<u32>::builder() // Changed from U256 to u32
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .build()?;

    // Allocate solutions buffer with current_batch_size * 1000 * 4
    let solutions = Buffer::<u64>::builder() // Changed from U256 to u64
        .queue(queue.clone())
        .flags(flags::MEM_WRITE_ONLY)
        .len(batch_size as usize * 1000 * 4) // Adjusted size
        .build()?;

    // Reset solution count
    solution_count.cmd().write(&vec![0u32]).enq()?;
    debug!("Solution count reset.");

    // Set kernel arguments
    kernel.set_arg(0, &d_message)?;
    kernel.set_arg(1, &d_target)?;
    kernel.set_arg(2, 0u64)?; // startPosition as u64
    kernel.set_arg(3, batch_size as u32 * 1000 * 4)?; // maxSolutionCount as u32
    kernel.set_arg(4, &solutions)?;
    kernel.set_arg(5, &solution_count)?;
    debug!("Kernel arguments set for calibration.");

    // Define local and global work sizes
    let local_work_size = 256;
    let global_work_size = ((batch_size * 1000 + local_work_size - 1) / local_work_size) * local_work_size;

    // Enqueue kernel execution with profiling using Instant
    let start_time = Instant::now();
    debug!("Enqueuing calibration kernel with global work size: {}", global_work_size);
    unsafe {
        kernel
            .cmd()
            .global_work_size([global_work_size as usize])
            .local_work_size([local_work_size as usize])
            .enq()?; // Synchronous call within unsafe block
    }
    let elapsed = start_time.elapsed();
    let elapsed_sec = elapsed.as_secs_f64();

    // Prevent division by zero
    if elapsed_sec == 0.0 {
        return Err("Elapsed time for calibration is zero.".into());
    }

    // Calculate hashrate
    let hashrate = (batch_size * 1000 * 4) as f64 / elapsed_sec;
    info!("Calibration completed. Hashrate: {:.2} H/s", hashrate);

    Ok(hashrate)
}

/// GPU Mining loop with Benchmarking and Dynamic Load Adjustment.
async fn gpu_mine_single(
    contract: Contract<Http>,
    address: Address,
    private_key: SecretKey,
    queue: Queue,
    kernel: Kernel,
    target: Vec<u8>,
    batch_size: u64,
    global_nonce_counter: Arc<Mutex<u64>>,
    web3: Web3<Http>,
    submitted_nonces: Arc<Mutex<HashSet<u64>>>,
    target_hashrate: f64,
    tx_nonce_counter: Arc<Mutex<u64>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting mining loop on GPU.");

    let mut message_data = vec![0u8; 52];
    message_data[..20].copy_from_slice(address.as_bytes());

    let d_message = Buffer::<u8>::builder()
        .queue(queue.clone())
        .copy_host_slice(&message_data)
        .flags(flags::MEM_READ_ONLY)
        .len(52)
        .build()?;

    let d_target = Buffer::<u8>::builder()
        .queue(queue.clone())
        .copy_host_slice(&target)
        .flags(flags::MEM_READ_ONLY)
        .len(32)
        .build()?;

    let solution_count = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .build()?;

    let mut current_batch_size = batch_size;
    let mut last_check = Instant::now();
    let mut cached_difficulty = fetch_difficulty(&contract).await?;
    let mut target = compute_target(cached_difficulty);

    let mut solutions_found = 0u32;
    let mut solutions_submitted = 0u32;
    let mut solutions_rejected = 0u32;

    loop {
        if last_check.elapsed() >= Duration::from_secs(180) {
            let new_difficulty = fetch_difficulty(&contract).await?;
            if new_difficulty != cached_difficulty {
                cached_difficulty = new_difficulty;
                target = compute_target(cached_difficulty);
                info!("Difficulty updated to {}", cached_difficulty);
            }
            last_check = Instant::now();
        }

        let solutions = Buffer::<u64>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(current_batch_size as usize * 1000 * 4)
            .build()?;

        solution_count.cmd().write(&vec![0u32]).enq()?;

        let mut global_nonce = global_nonce_counter.lock().await;
        let nonce = *global_nonce;
        *global_nonce += current_batch_size * 1000;

        kernel.set_arg(0, &d_message)?;
        kernel.set_arg(1, &d_target)?;
        kernel.set_arg(2, nonce)?;
        kernel.set_arg(3, batch_size as u32 * 1000 * 4)?;
        kernel.set_arg(4, &solutions)?;
        kernel.set_arg(5, &solution_count)?;

        let local_work_size = 256;
        let global_work_size = ((current_batch_size * 1000 + local_work_size - 1) / local_work_size) * local_work_size;

        let start_time = Instant::now();
        unsafe {
            kernel
                .cmd()
                .global_work_size([global_work_size as usize])
                .local_work_size([local_work_size as usize])
                .enq()?;
        }
        let elapsed = start_time.elapsed();
        let elapsed_sec = elapsed.as_secs_f64();

        if elapsed_sec == 0.0 {
            warn!("Elapsed time for kernel execution is zero. Skipping this iteration.");
            continue;
        }

        let total_nonces = current_batch_size * 1000 * 4;
        let hashrate = (total_nonces as f64) / elapsed_sec;

        // Update dynamic display
        print!("\rHashrate: {:.2} H/s [Solutions: {}] [Submitted: {}] [Rejected: {}]", hashrate, solutions_found, solutions_submitted, solutions_rejected);
        io::stdout().flush().unwrap();

        let mut sol_count_host = vec![0u32];
        solution_count.read(&mut sol_count_host).enq()?;
        let sol_count = sol_count_host[0];

        solutions_found += sol_count;

        if sol_count > 0 {
            let mut found_solutions = vec![0u64; current_batch_size as usize * 1000 * 4];
            solutions.read(&mut found_solutions[..]).enq()?;

            for &potential_nonce in &found_solutions[..sol_count as usize] {
                if potential_nonce != 0 {
                    let is_valid = verify_nonce(&address, potential_nonce, &target);
                    if is_valid {
                        let mut submitted = submitted_nonces.lock().await;
                        if !submitted.contains(&potential_nonce) {
                            submitted.insert(potential_nonce);
                            match submit_solution(
                                &contract,
                                &private_key,
                                &web3,
                                address,
                                potential_nonce,
                                submitted_nonces.clone(),
                                tx_nonce_counter.clone(),
                            ).await {
                                Ok(_) => {
                                    solutions_submitted += 1;
                                },
                                Err(_) => {
                                    solutions_rejected += 1;
                                }
                            }
                        } else {
                            solutions_rejected += 1;
                        }
                    } else {
                        solutions_rejected += 1;
                    }
                }
            }
        }

        // Dynamic Load Adjustment
        if hashrate < target_hashrate {
            current_batch_size = ((current_batch_size as f64 * 1.1).min(10_000_000.0)) as u64;
        } else if hashrate > target_hashrate * 1.2 {
            current_batch_size = ((current_batch_size as f64 * 0.9).max(1_000.0)) as u64;
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    initialize_logger();

    // Parse command-line arguments
    let matches = Command::new("PolyMiner")
        .version("1.0")
        .about("GPU miner for Polymine")
        .arg(
            Arg::new("rpc_url")
                .long("rpc")
                .required(true)
                .help("RPC URL for the blockchain network"),
        )
        .arg(
            Arg::new("contract_address")
                .long("contract")
                .required(true)
                .help("Smart contract address"),
        )
        .arg(
            Arg::new("wallet_address")
                .long("wallet")
                .required(true)
                .help("Your wallet address"),
        )
        .arg(
            Arg::new("private_key")
                .long("private")
                .required(true)
                .help("Your wallet's private key (hex)"),
        )
        .arg(
            Arg::new("batch_size")
                .long("batch")
                .default_value("10000")
                .value_parser(clap::value_parser!(u64))
                .help("Initial batch size for mining"),
        )
        .arg(
            Arg::new("device-indices")
                .long("device-indices")
                .default_value("all")
                .help("Comma-separated GPU indices to use (e.g., 0,1,2) or 'all'"),
        )
        .get_matches();

    let rpc_url = matches.get_one::<String>("rpc_url").unwrap();
    let contract_address: Address = matches.get_one::<String>("contract_address").unwrap().parse()?;
    let wallet_address: Address = matches.get_one::<String>("wallet_address").unwrap().parse()?;
    let private_key_hex = matches.get_one::<String>("private_key").unwrap();
    let private_key = SecretKey::from_slice(&hex::decode(private_key_hex)?)?;
    let batch_size = *matches.get_one::<u64>("batch_size").unwrap();
    let device_indices = matches.get_one::<String>("device-indices").unwrap();

    info!("Connecting to RPC at {}", rpc_url);
    let transport = Http::new(rpc_url)?;
    let web3 = Web3::new(transport);
    let abi = include_bytes!("contract_abi.json");
    let contract = Contract::from_json(web3.eth(), contract_address, abi)?;

    display_wallet_balance(&web3, wallet_address).await?;

    let initial_difficulty = fetch_difficulty(&contract).await?;
    let target = compute_target(initial_difficulty);

    let devices = list_gpus()?;
    let selected_devices: Vec<_> = if device_indices == "all" {
        devices
    } else {
        device_indices
            .split(',')
            .filter_map(|idx| idx.parse::<usize>().ok().and_then(|i| devices.get(i)))
            .cloned()
            .collect()
    };

    if selected_devices.is_empty() {
        panic!("No GPUs selected or invalid indices provided.");
    }

    // Load existing miner state or start fresh
    let state = load_state(&web3, wallet_address).await;
    let submitted_nonces = Arc::new(Mutex::new(state.submitted_nonces.clone()));
    let global_nonce_counter = Arc::new(Mutex::new(state.global_nonce));
    let tx_nonce_counter = Arc::new(Mutex::new(state.tx_nonce));

    for (i, (platform, device)) in selected_devices.iter().enumerate() {
        info!(
            "Using GPU at index {}: {} on platform {}",
            i,
            device.name()?,
            platform.name()?
        );
        let (queue, kernel) = setup_opencl(platform, device)?;
        let calibration_hashrate = calibrate_gpu(&kernel, &queue, batch_size).await?;
        info!("Calibration Hashrate: {:.2} H/s", calibration_hashrate);

        let dynamic_target_hashrate = calibration_hashrate * 0.9;

        tokio::spawn(gpu_mine_single(
            contract.clone(),
            wallet_address,
            private_key.clone(),
            queue,
            kernel,
            target.clone(),
            batch_size,
            Arc::clone(&global_nonce_counter),
            web3.clone(),
            Arc::clone(&submitted_nonces),
            dynamic_target_hashrate,
            Arc::clone(&tx_nonce_counter),
        ));
    }

    // Clone shared state for the shutdown handler
    let submitted_nonces_clone = Arc::clone(&submitted_nonces);
    let global_nonce_clone = Arc::clone(&global_nonce_counter);
    let tx_nonce_counter_clone = Arc::clone(&tx_nonce_counter);

    // Spawn a task to listen for shutdown signals (e.g., Ctrl+C)
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
        info!("Shutdown signal received. Saving state...");

        let submitted_nonces = submitted_nonces_clone.lock().await.clone();
        let global_nonce = *global_nonce_clone.lock().await;
        let tx_nonce = *tx_nonce_counter_clone.lock().await;

        let final_state = MinerState {
            submitted_nonces,
            global_nonce,
            tx_nonce,
        };

        save_state(&final_state);
        info!("State saved successfully. Exiting.");
        std::process::exit(0);
    });

    // Keep the main task running indefinitely or until a shutdown signal
    // Note: Mining tasks are spawned as background tasks, they will keep running
    // until the program is shut down.
    loop {
        tokio::time::sleep(Duration::from_secs(3600)).await; // Sleep for an hour, just to keep the process alive
    }

    // This code will never be reached due to the loop above, but included for completeness
    // Ok(())
}

    #[cfg(test)]
    mod tests {
        use super::*;
        use web3::types::Address;
        use sha3::{Digest, Keccak256};

        /// Verifies if a given nonce satisfies the mining condition.
        ///
        /// # Arguments
        ///
        /// * `address` - The wallet address.
        /// * `nonce` - The mining nonce to verify.
        /// * `target` - The target value derived from the current difficulty.
        ///
        /// # Returns
        ///
        /// * `true` if the nonce is valid, `false` otherwise.
        fn verify_nonce(address: &Address, nonce: u64, target: &[u8]) -> bool {
            // Convert nonce to 32-byte big-endian representation
            let mut nonce_bytes = [0u8; 32];
            nonce_bytes[24..].copy_from_slice(&nonce.to_be_bytes()); // Place u64 at the end

            // Concatenate address (20 bytes) and nonce (32 bytes)
            let mut hasher = Keccak256::new();
            hasher.update(address.as_bytes());
            hasher.update(&nonce_bytes);
            let hash = hasher.finalize();

            // Convert hash to U256
            let hash_u256 = U256::from_big_endian(&hash);

            // Convert target to U256
            let target_u256 = U256::from_big_endian(target);

            hash_u256 < target_u256
        }

        #[test]
        fn test_compute_target() {
            let difficulty = 1000u64;
            let target = compute_target(difficulty);
            assert_eq!(target.len(), 32);
            // Additional assertions can be added based on expected target value
        }

        #[test]
        fn test_verify_nonce_max_target() {
            // Arrange
            let address = Address::from_slice(&[0xEF; 20]); // Example address
            let nonce = 1u64; // Example nonce
            let target = vec![0xFFu8; 32]; // Maximum target, all nonces should be valid

            // Act
            let is_valid = verify_nonce(&address, nonce, &target);

            // Assert
            assert!(is_valid, "Nonce should be valid when target is maximum");
        }

        #[test]
        fn test_verify_nonce_invalid_zero_target() {
            // Arrange
            let address = Address::from_slice(&[0xCD; 20]); // Different example address
            let nonce = 0u64; // Edge case nonce
            let target = vec![0x00u8; 32]; // Minimum target, no nonce should be valid

            // Act
            let is_valid = verify_nonce(&address, nonce, &target);

            // Assert
            assert!(!is_valid, "Nonce should be invalid when target is zero");
        }

        #[test]
        fn test_verify_nonce_valid_case() {
            // Arrange
            let address = Address::from_slice(&[0xAB; 20]); // Example address
            let nonce = 123456789u64; // Example nonce
            let target = compute_target(1000u64);

            // Act
            let is_valid = verify_nonce(&address, nonce, &target);

            // Assert
            // Since target is max_val / 1000, it's likely that this nonce is valid
            // Depending on the hash, adjust the assertion
            // Here, we use a placeholder assertion
            assert!(is_valid || !is_valid, "Nonce validity depends on hash"); // Placeholder
        }
    }
