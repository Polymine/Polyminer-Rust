use clap::{Arg, Command};
use ocl::{Buffer, Device, Kernel, Platform, Program, Queue};
use ocl::builders::ContextBuilder;
use serde::Deserialize;
use sha3::{Digest, Keccak256};
use web3::contract::{Contract, Options};
use web3::types::{Address, U256};
use web3::transports::Http;
use web3::Web3;
use web3::signing::SecretKey;
use hex;
use tokio;
use generic_array::GenericArray;

const KERNEL_SOURCE: &str = include_str!("kernel.cl");

#[derive(Deserialize)]
struct Config {
    rpc_url: String,
    contract_address: String,
    private_key: String,
    batch_size: u64,
}

/// Fetches the current mining difficulty from the smart contract.
async fn fetch_difficulty(
    contract: &Contract<Http>,
) -> Result<U256, Box<dyn std::error::Error + Send + Sync>> {
    let difficulty: U256 = contract.query("difficulty", (), None, Options::default(), None).await?;
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

/// Displays the wallet's balance.
async fn display_wallet_balance(web3: &Web3<Http>, wallet_address: Address) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let balance = web3.eth().balance(wallet_address, None).await?;
    println!("Wallet balance: {} MATIC", balance / U256::exp10(18));
    Ok(())
}

/// Submits a valid nonce to the smart contract.
async fn submit_solution(
    contract: &Contract<Http>,
    private_key: SecretKey,
    nonce: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let options = Options {
        value: Some(U256::exp10(16)), // 0.01 MATIC in Wei
        ..Default::default()
    };
    let tx_hash = contract
        .signed_call("mine", (nonce,), options, &private_key)
        .await?;
    println!("Submitted nonce {} with TX: {:?}", nonce, tx_hash);
    Ok(())
}

/// Lists all available GPU devices.
fn list_gpus() -> Result<Vec<Device>, Box<dyn std::error::Error + Send + Sync>> {
    let platforms = Platform::list();
    let mut devices = Vec::new();
    for platform in platforms {
        devices.extend(Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU))?);
    }
    Ok(devices)
}

/// Sets up the OpenCL queue and kernel.
fn setup_opencl(device: &Device) -> Result<(Queue, Kernel), Box<dyn std::error::Error + Send + Sync>> {
    let context = ContextBuilder::new()
        .devices(*device)
        .build()?;

    let queue = Queue::new(&context, *device, None)?;
    let program = Program::builder()
        .src(KERNEL_SOURCE)
        .devices(*device)
        .build(&context)?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("hashMessage")
        .queue(queue.clone())
        .global_work_size(1)
        .arg(None::<&Buffer<u8>>)
        .arg(None::<&Buffer<u8>>)
        .arg(0u64)
        .arg(0u32)
        .arg(None::<&Buffer<u64>>)
        .arg(None::<&Buffer<u32>>)
        .build()?;

    Ok((queue, kernel))
}

/// GPU Mining loop for a single GPU.
async fn gpu_mine_single(
    contract: Contract<Http>,
    address: Address,
    private_key: SecretKey,
    queue: Queue,
    kernel: Kernel,
    target: Vec<u8>,
    batch_size: u64,
    start_nonce: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut message_data = vec![0u8; 52];
    message_data[..20].copy_from_slice(address.as_bytes());

    let d_message = Buffer::<u8>::builder()
        .queue(queue.clone())
        .copy_host_slice(&message_data)
        .len(52)
        .build()?;

    let d_target = Buffer::<u8>::builder()
        .queue(queue.clone())
        .copy_host_slice(&target)
        .len(32)
        .build()?;

    let solutions = Buffer::<u64>::builder()
        .queue(queue.clone())
        .len(batch_size as usize)
        .fill_val(0u64)
        .build()?;

    let solution_count = Buffer::<u32>::builder()
        .queue(queue.clone())
        .len(1)
        .fill_val(0u32)
        .build()?;

    let mut nonce = start_nonce;

    loop {
        solution_count.cmd().write(&vec![0u32]).enq()?;
        kernel.set_arg(0, &d_message)?;
        kernel.set_arg(1, &d_target)?;
        kernel.set_arg(2, &nonce)?;
        kernel.set_arg(3, &(batch_size as u32))?;
        kernel.set_arg(4, &solutions)?;
        kernel.set_arg(5, &solution_count)?;

        unsafe {
            kernel.cmd()
                .global_work_size([batch_size as usize])
                .enq()?;
        }

        let mut found_solutions = vec![0u64; batch_size as usize];
        solutions.read(&mut found_solutions).enq()?;

        let mut sol_count_host = vec![0u32];
        solution_count.read(&mut sol_count_host).enq()?;
        let sol_count = sol_count_host[0];

        if sol_count > 0 {
            println!("Found {} potential solution(s)", sol_count);
            for &potential_nonce in &found_solutions[..sol_count as usize] {
                if verify_nonce(&address, potential_nonce, &target) {
                    println!("Verified nonce: {}", potential_nonce);
                    submit_solution(&contract, private_key.clone(), potential_nonce).await?;
                    return Ok(());
                }
            }
        }

        nonce += batch_size * 2; // Increment nonce for this GPU
    }
}

/// Verifies the nonce on the CPU.
fn verify_nonce(
    address: &Address,
    nonce: u64,
    target: &[u8],
) -> bool {
    let mut message = Vec::new();
    message.extend_from_slice(address.as_bytes());

    let mut nonce_bytes = [0u8; 32];
    nonce_bytes[24..].copy_from_slice(&nonce.to_be_bytes());
    message.extend_from_slice(&nonce_bytes);

    let hash = Keccak256::digest(&message);
    let target_array = GenericArray::clone_from_slice(target);
    hash < target_array
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let matches = Command::new("PolyMiner")
        .version("1.0")
        .arg(Arg::new("rpc_url").long("rpc").required(true))
        .arg(Arg::new("contract_address").long("contract").required(true))
        .arg(Arg::new("wallet_address").long("wallet").required(true))
        .arg(Arg::new("private_key").long("private").required(true))
        .arg(Arg::new("batch_size").long("batch").default_value("1000000"))
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
    let batch_size = matches.get_one::<String>("batch_size").unwrap().parse::<u64>().unwrap();
    let device_indices = matches.get_one::<String>("device-indices").unwrap();

    let web3 = Web3::new(Http::new(rpc_url)?);
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
    for (i, device) in selected_devices.iter().enumerate() {
        println!("Using GPU at index {}: {}", i, device.name()?);
        let (queue, kernel) = setup_opencl(device)?;
        let contract = contract.clone();
        let address = wallet_address;
        let private_key = private_key.clone();
        let target = target.clone();
        let start_nonce = i as u64 * batch_size;

        handles.push(tokio::spawn(async move {
            gpu_mine_single(contract, address, private_key, queue, kernel, target, batch_size, start_nonce).await
        }));
    }

    for handle in handles {
        handle.await??;
    }

    Ok(())
}
