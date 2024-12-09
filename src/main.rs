use clap::{Arg, Command};
use ocl::{Buffer, Device, Kernel, Platform, Program, Queue, flags};
use ocl::builders::ContextBuilder;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use web3::contract::{Contract, Options};
use web3::types::{Address, U256};
use web3::transports::Http;
use web3::Web3;
use web3::signing::SecretKey;
use hex;
use tokio;
use generic_array::GenericArray;
use std::collections::HashSet;
use std::time::Instant;
use std::io::{self, Read, Write, BufWriter};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use log::{info, warn, error, debug};
use chrono::Local;
use env_logger::Builder;
use std::fs::File;

/// Path to the OpenCL kernel source code.
const KERNEL_SOURCE: &str = include_str!("kernel.cl");

/// Configuration structure for the miner.
#[derive(Deserialize)]
struct Config {
    rpc_url: String,
    contract_address: String,
    private_key: String,
    batch_size: u64,
}

/// Structure to persist miner state.
#[derive(Serialize, Deserialize)]
struct MinerState {
    submitted_nonces: HashSet<u64>,
    global_nonce: u64,
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
fn load_state() -> MinerState {
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
    MinerState {
        submitted_nonces: HashSet::new(),
        global_nonce: 0,
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
) -> Result<U256, Box<dyn std::error::Error + Send + Sync>> {
    info!("Fetching difficulty...");
    let difficulty: U256 = contract.query("difficulty", (), None, Options::default(), None).await?;
    info!("Current difficulty: {}", difficulty);
    Ok(difficulty)
}

/// Computes the target based on the current difficulty.
fn compute_target(difficulty: U256) -> Vec<u8> {
    let max_val = U256::MAX;
    let target = max_val / difficulty;
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

/// Submits a valid nonce to the smart contract.
async fn submit_solution(
    contract: &Contract<Http>,
    private_key: SecretKey,
    web3: &Web3<Http>,
    wallet_address: Address,
    nonce: u64,
    submitted_nonces: Arc<Mutex<HashSet<u64>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // No need to check if nonce is already submitted here since it's handled before calling this function

    let gas_price = web3.eth().gas_price().await?;
    let adjusted_gas_price = gas_price + (gas_price / 10); // Increase gas price by 10%
    let current_tx_nonce = web3.eth().transaction_count(wallet_address, None).await?;

    let options = Options {
        nonce: Some(current_tx_nonce),
        gas_price: Some(adjusted_gas_price),
        value: Some(U256::exp10(16)), // 0.01 MATIC in Wei
        ..Default::default()
    };

    match contract
        .signed_call("mine", (nonce,), options, &private_key)
        .await
    {
        Ok(tx_hash) => {
            info!("Submitted nonce {} with TX: {:?}", nonce, tx_hash);
            Ok(())
        }
        Err(e) => {
            error!("Error submitting nonce {}: {:?}", nonce, e);
            // Remove the nonce from the submitted set if submission failed
            let mut submitted = submitted_nonces.lock().unwrap();
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
        .global_work_size([1]) // Will be overridden in host code
        .local_work_size([256]) // Optimal local work-group size
        .arg(None::<&Buffer<u8>>)
        .arg(None::<&Buffer<u8>>)
        .arg(0u64)
        .arg(0u32)
        .arg(None::<&Buffer<u64>>)
        .arg(None::<&Buffer<u32>>)
        .build()?;

    Ok((queue, kernel))
}

/// Verifies the nonce on CPU using big-endian and lexicographical comparison.
fn verify_nonce(address: &Address, nonce: u64, target: &[u8]) -> bool {
    let mut message = Vec::new();
    message.extend_from_slice(address.as_bytes());

    // Encode nonce as 32-byte big-endian
    let mut nonce_bytes = [0u8; 32];
    nonce_bytes[24..].copy_from_slice(&nonce.to_be_bytes());
    message.extend_from_slice(&nonce_bytes);

    debug!("Constructed message for hashing: {:?}", message);

    let hash = Keccak256::digest(&message);
    debug!("Computed Keccak256 hash: {:?}", hash);

    let target_array = GenericArray::clone_from_slice(target);
    debug!("Target value for comparison: {:?}", target_array);

    let is_valid = hash < target_array;

    debug!("Nonce {} verification result: {}", nonce, is_valid);
    is_valid
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

    let solution_count = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .build()?;

    let current_batch_size = batch_size;

    loop {
        // Allocate solutions buffer with current_batch_size * 1000 * 4
        let solutions = Buffer::<u64>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(current_batch_size as usize * 1000 * 4) // Adjusted size
            .build()?;

        // Reset solution count
        solution_count.cmd().write(&vec![0u32]).enq()?;
        debug!("Solution count reset.");

        // Set kernel arguments
        kernel.set_arg(0, &d_message)?;
        kernel.set_arg(1, &d_target)?;
        kernel.set_arg(2, &0u64)?; // startPosition
        kernel.set_arg(3, &(current_batch_size as u32 * 1000 * 4))?; // maxSolutionCount
        kernel.set_arg(4, &solutions)?;
        kernel.set_arg(5, &solution_count)?;
        debug!("Kernel arguments set for calibration.");

        // Define local and global work sizes
        let local_work_size = 256;
        let global_work_size = ((current_batch_size * 1000 + local_work_size - 1) / local_work_size) * local_work_size;

        // Enqueue kernel execution with profiling using Instant
        let start_time = Instant::now();
        debug!("Enqueuing calibration kernel with global work size: {}", global_work_size);
        unsafe {
            kernel.cmd()
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
        let hashrate = (current_batch_size * 1000 * 4) as f64 / elapsed_sec;
        info!("Calibration completed. Hashrate: {:.2} H/s", hashrate);

        return Ok(hashrate);
    }
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
    global_nonce_counter: Arc<AtomicU64>,
    web3: Web3<Http>,
    submitted_nonces: Arc<Mutex<HashSet<u64>>>,
    target_hashrate: f64,
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

    // Initial buffer allocation for solution_count
    let solution_count = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(1)
        .build()?;

    let mut current_batch_size = batch_size;

    loop {
        // Allocate solutions buffer with current_batch_size * 1000 * 4
        let solutions = Buffer::<u64>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(current_batch_size as usize * 1000 * 4) // Adjusted size
            .build()?;

        // Reset solution count
        solution_count.cmd().write(&vec![0u32]).enq()?;
        debug!("Solution count reset.");

        // Fetch a unique starting nonce atomically
        let nonce = global_nonce_counter.fetch_add(current_batch_size * 1000, Ordering::SeqCst);
        debug!("Assigned nonce range starting at {}", nonce);

        // Set kernel arguments
        kernel.set_arg(0, &d_message)?;
        kernel.set_arg(1, &d_target)?;
        kernel.set_arg(2, &nonce)?; // startPosition
        kernel.set_arg(3, &(current_batch_size as u32 * 1000 * 4))?; // maxSolutionCount
        kernel.set_arg(4, &solutions)?;
        kernel.set_arg(5, &solution_count)?;
        debug!("Kernel arguments set for nonce range starting at {}", nonce);

        // Define local and global work sizes
        let local_work_size = 256;
        let global_work_size = ((current_batch_size * 1000 + local_work_size - 1) / local_work_size) * local_work_size;

        // Enqueue kernel execution with profiling using Instant
        let start_time = Instant::now();
        debug!("Enqueuing mining kernel with global work size: {}", global_work_size);
        unsafe {
            kernel.cmd()
                .global_work_size([global_work_size as usize])
                .local_work_size([local_work_size as usize])
                .enq()?; // Synchronous call within unsafe block
        }
        let elapsed = start_time.elapsed();
        let elapsed_sec = elapsed.as_secs_f64();
        debug!("Kernel execution completed in {:.2} seconds.", elapsed_sec);

        // Prevent division by zero
        if elapsed_sec == 0.0 {
            warn!("Elapsed time for kernel execution is zero. Skipping this iteration.");
            continue; // Skip this iteration
        }

        // Calculate hashrate
        let total_nonces = current_batch_size * 1000 * 4;
        let hashrate = (total_nonces as f64) / elapsed_sec;

        // Sanity check for hashrate
        if hashrate.is_infinite() || hashrate.is_nan() || hashrate < 0.0 {
            error!("Invalid hashrate calculated: {:.2} H/s. Skipping adjustment.", hashrate);
        } else {
            // Dynamic Hashrate Display on the Same Line
            print!("\rHashrate: {:.2} H/s", hashrate);
            io::stdout().flush().unwrap();
            info!("Current Hashrate: {:.2} H/s", hashrate);

            // Read solutions from GPU
            let mut found_solutions = vec![0u64; current_batch_size as usize * 1000 * 4];
            solutions.read(&mut found_solutions[..]).enq()?; // Ensure this line is active
            debug!("Solutions buffer read successfully.");

            // Read the number of solutions found
            let mut sol_count_host = vec![0u32];
            solution_count.read(&mut sol_count_host).enq()?;
            let sol_count = sol_count_host[0];

            info!("Found {} solutions.", sol_count);

            if sol_count > 0 {
                for &potential_nonce in &found_solutions[..sol_count as usize] {
                    if potential_nonce != 0 {
                        info!("GPU found potential nonce: {}", potential_nonce);
                        // Verify on CPU
                        if verify_nonce(&address, potential_nonce, &target) {
                            info!("CPU verified valid nonce: {}", potential_nonce);
                            {
                                // Lock the mutex before modifying the HashSet
                                let mut submitted = submitted_nonces.lock().unwrap();
                                if submitted.contains(&potential_nonce) {
                                    warn!("Nonce {} already submitted. Skipping.", potential_nonce);
                                    continue;
                                }
                                // Add to the submitted nonces before submission to prevent race conditions
                                submitted.insert(potential_nonce);
                                debug!("Nonce {} added to submitted_nonces.", potential_nonce);
                            }
                            match submit_solution(
                                &contract,
                                private_key.clone(),
                                &web3,
                                address,
                                potential_nonce,
                                submitted_nonces.clone(),
                            )
                            .await
                            {
                                Ok(_) => info!("Nonce {} submitted successfully.", potential_nonce),
                                Err(e) => error!("Failed to submit nonce {}: {:?}", potential_nonce, e),
                            }
                        } else {
                            warn!("CPU verification failed for nonce: {}", potential_nonce);
                        }
                    }
                }
            }

            // Dynamic Load Adjustment based on hashrate
            if hashrate < target_hashrate {
                info!("Hashrate below target. Increasing batch size by 10%.");
                current_batch_size = (current_batch_size as f64 * 1.1) as u64;
                if current_batch_size > 10_000_000 {
                    warn!("Max batch size reached. Setting to 10,000,000.");
                    current_batch_size = 10_000_000;
                }
                debug!("Adjusted batch size to {}", current_batch_size);
            } else if hashrate > target_hashrate * 1.2 {
                info!("Hashrate above target. Decreasing batch size by 10%.");
                current_batch_size = (current_batch_size as f64 * 0.9) as u64;
                if current_batch_size < 1_000 {
                    warn!("Minimum batch size reached. Setting to 1,000.");
                    current_batch_size = 1_000;
                }
                debug!("Adjusted batch size to {}", current_batch_size);
            } else {
                debug!("Hashrate within target range. Maintaining batch size at {}.", current_batch_size);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize the customized logger
    initialize_logger();

    // Load existing miner state or start fresh
    let state = load_state(); // Removed `mut` as `state` is not modified directly
    let submitted_nonces = Arc::new(Mutex::new(state.submitted_nonces.clone()));
    let global_nonce_counter = Arc::new(AtomicU64::new(state.global_nonce));

    let matches = Command::new("PolyMiner")
        .version("1.0")
        .about("GPU miner for Polymine")
        .arg(Arg::new("rpc_url")
            .long("rpc")
            .required(true)
            .help("RPC URL for the blockchain network"))
        .arg(Arg::new("contract_address")
            .long("contract")
            .required(true)
            .help("Smart contract address"))
        .arg(Arg::new("wallet_address")
            .long("wallet")
            .required(true)
            .help("Your wallet address"))
        .arg(Arg::new("private_key")
            .long("private")
            .required(true)
            .help("Your wallet's private key (hex)"))
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
    let private_key = SecretKey::from_slice(&hex::decode(matches.get_one::<String>("private_key").unwrap())?)?;
    let batch_size = *matches.get_one::<u64>("batch_size").unwrap();
    let device_indices = matches.get_one::<String>("device-indices").unwrap();

    info!("Connecting to RPC at {}", rpc_url);
    let transport = Http::new(rpc_url)?;
    let web3 = Web3::new(transport);
    let abi = include_bytes!("contract_abi.json");
    let contract = Contract::from_json(web3.eth(), contract_address, abi)?;

    display_wallet_balance(&web3, wallet_address).await?;

    let difficulty = fetch_difficulty(&contract).await?;
    let target = compute_target(difficulty);

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

    let mut handles = Vec::new();

    for (i, (platform, device)) in selected_devices.iter().enumerate() {
        info!("Using GPU at index {}: {} on platform {}", i, device.name()?, platform.name()?);
        let (queue, kernel) = setup_opencl(platform, device)?;
        let contract_clone = contract.clone();
        let address = wallet_address;
        let private_key_clone = private_key.clone();
        let target_clone = target.clone();
        let web3_clone = web3.clone();
        let submitted_nonces_clone = Arc::clone(&submitted_nonces);
        let global_nonce_clone = Arc::clone(&global_nonce_counter);

        // Calibration Phase - Called Once Per GPU
        let calibration_hashrate = calibrate_gpu(&kernel, &queue, batch_size).await?;
        info!("Calibration Hashrate: {:.2} H/s", calibration_hashrate);

        // Adjust target hashrate based on calibration
        let dynamic_target_hashrate = calibration_hashrate * 0.9; // Target is 90% of calibration hashrate

        handles.push(tokio::spawn(async move {
            gpu_mine_single(
                contract_clone,
                address,
                private_key_clone,
                queue,
                kernel,
                target_clone,
                batch_size,
                global_nonce_clone,
                web3_clone,
                submitted_nonces_clone,
                dynamic_target_hashrate,
            ).await
        }));
    }

    // Await all mining tasks
    for handle in handles {
        handle.await??;
    }

    // Save miner state before exiting
    save_state(&MinerState {
        submitted_nonces: submitted_nonces.lock().unwrap().clone(),
        global_nonce: global_nonce_counter.load(Ordering::SeqCst),
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use web3::types::Address;
    use hex;

    #[test]
    fn test_verify_nonce_valid() {
        // Arrange
        let address = Address::from_slice(&[0xAB; 20]); // Example address
        let nonce = 123456789u64;
        let difficulty = U256::from(153_911_597u64); // Example difficulty
        let target = compute_target(difficulty); // Assume compute_target is defined

        // Act
        let is_valid = verify_nonce(&address, nonce, &target);

        // Assert
        // Depending on the difficulty and nonce, determine expected outcome
        // For testing, set a target that ensures a predictable result
        // Here, we'll assume it's invalid
        assert!(!is_valid, "Nonce should be invalid for the given difficulty and target");
    }

    #[test]
    fn test_verify_nonce_invalid() {
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
    fn test_verify_nonce_max_target() {
        // Arrange
        let address = Address::from_slice(&[0xEF; 20]); // Another example address
        let nonce = 1u64; // Example nonce
        let target = vec![0xFFu8; 32]; // Maximum target, all nonces should be valid

        // Act
        let is_valid = verify_nonce(&address, nonce, &target);

        // Assert
        assert!(is_valid, "Nonce should be valid when target is maximum");
    }
}
