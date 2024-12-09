# PolyMiner README

## **PolyMiner: A Rust-based GPU Miner**

PolyMiner is a high-performance GPU miner written in Rust. This guide explains how to set up your environment, build PolyMiner from the source, and run it on your system.

---

## **Table of Contents**

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Building PolyMiner](#building-polyminer)
4. [Running PolyMiner](#running-polyminer)
5. [Command-Line Arguments](#command-line-arguments)
6. [Example Usage](#example-usage)
7. [Troubleshooting](#troubleshooting)

---

## **Prerequisites**

To build and run PolyMiner, ensure you have the following installed on your system:

### **1. Rust Toolchain**
- Install Rust using `rustup`:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- After installation, ensure Rust and Cargo (Rust's package manager) are available:
  ```bash
  rustc --version
  cargo --version
  ```

### **2. OpenCL Drivers**
- Install OpenCL drivers for your GPU:
  - **AMD GPUs**: Download and install from the [AMD Drivers & Support page](https://www.amd.com/en/support).
  - **NVIDIA GPUs**: Ensure CUDA drivers are installed. OpenCL support is included with CUDA.
  - **Intel GPUs**: Install the Intel OpenCL runtime.

### **3. System Tools**
- Ensure `git` is installed to clone the repository:
  ```bash
  sudo apt install git  # On Ubuntu/Debian
  brew install git      # On macOS
  choco install git     # On Windows using Chocolatey
  ```

---

## **Installation**

Clone the PolyMiner repository to your local system:

```bash
git clone https://github.com/Polyminer/Polyminer-Rust.git
cd Polyminer-Rust
```

---

## **Building PolyMiner**

1. **Ensure Dependencies Are Installed:**
   Install the required Rust dependencies listed in `Cargo.toml`:
   ```bash
   cargo build
   ```

2. **Build the Project:**
   Build the release version of PolyMiner for optimal performance:
   ```bash
   cargo build --release
   ```

3. **Verify the Build:**
   The compiled binary will be located in the `target/release` directory:
   ```bash
   ls target/release/polyminer
   ```

---

## **Running PolyMiner**

To run PolyMiner, use the compiled binary with appropriate arguments. For example:

```bash
./target/release/polyminer --rpc <RPC_URL> --contract <CONTRACT_ADDRESS> --wallet <WALLET_ADDRESS> --private <PRIVATE_KEY> --batch <BATCH_SIZE> --device-indices <DEVICE_INDICES>
```

---

## **Command-Line Arguments**

| Argument          | Description                                                                                          | Required | Example                              |
|--------------------|------------------------------------------------------------------------------------------------------|----------|--------------------------------------|
| `--rpc`           | RPC URL of the blockchain network.                                                                  | Yes      | `https://polygon-mainnet.rpc.com`   |
| `--contract`      | Smart contract address for mining.                                                                  | Yes      | `0x1234567890abcdef...`             |
| `--wallet`        | Wallet address for receiving mining rewards.                                                        | Yes      | `0xabcdef1234567890...`             |
| `--private`       | Private key of the wallet in hexadecimal format.                                                    | Yes      | `abcdef1234567890...`               |
| `--batch`         | Initial batch size for mining (adjusts dynamically).                                                | No       | `10000`                             |
| `--device-indices`| Comma-separated GPU indices to use (e.g., `0,1,2`) or `all` to use all GPUs.                        | No       | `0,1` or `all`                      |

---

## **Example Usage**

### **Single GPU**
Run PolyMiner on a single GPU with a batch size of 10,000:
```bash
./target/release/polyminer \
  --rpc https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY \
  --contract 0xYourContractAddress \
  --wallet 0xYourWalletAddress \
  --private YourPrivateKeyInHex \
  --batch 10000 \
  --device-indices 0
```

### **Multiple GPUs**
Run PolyMiner on all available GPUs:
```bash
./target/release/polyminer \
  --rpc https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY \
  --contract 0xYourContractAddress \
  --wallet 0xYourWalletAddress \
  --private YourPrivateKeyInHex \
  --batch 20000 \
  --device-indices all
```

---

## **Troubleshooting**

1. **Missing OpenCL Drivers**
   - Ensure GPU drivers are correctly installed.
   - Verify OpenCL support using the `clinfo` command:
     ```bash
     clinfo
     ```

2. **Build Errors**
   - If you encounter build errors, try updating Rust and Cargo:
     ```bash
     rustup update
     ```

3. **Runtime Errors**
   - Ensure the contract ABI file (`contract_abi.json`) is in the same directory as the executable.

4. **Performance Issues**
   - Monitor GPU utilization using tools like:
     - **NVIDIA**: `nvidia-smi`
     - **AMD**: `rocm-smi`

---

## **Development Notes**

1. **Modify OpenCL Kernel Code**
   - The OpenCL kernel code is located in `src/kernel.cl`.
   - Make changes to improve performance or customize hash functions.

2. **Testing**
   - Run tests using the following command:
     ```bash
     cargo test
     ```

3. **Contributing**
   - Contributions are welcome! Open a pull request or report issues via GitHub.

---

## **License**

PolyMiner is open-source software licensed under the MIT License. See the `LICENSE` file for more details.
