
// Define a structure for nonce vectors
typedef struct {
    ulong n0;
    ulong n1;
    ulong n2;
    ulong n3;
} nonce_vec_t;

// Compare two byte arrays lexicographically
static inline int compare(const uchar *left, const uchar *right, uint len) {
    for(uint i = 0; i < len; i++) {
        if(left[i] < right[i]) return -1;
        if(left[i] > right[i]) return 1;
    }
    return 0;
}

// Keccak-f permutation constants
#define KECCAK_ROUNDS 24
__constant uchar keccak_rotc[KECCAK_ROUNDS] = {
    1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 
    2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 
    39, 61, 20, 44
};

__constant uchar keccak_piln[KECCAK_ROUNDS] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 
    24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 
    22, 9, 6, 1
};

// Rotate left macro
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

// Keccak-f permutation
void keccakf(ulong state[25], const uchar rounds) {
    uchar r;
    for(r = 0; r < rounds; r++) {
        // Theta step
        ulong C[5];
        for(int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        ulong D[5];
        for(int x = 0; x < 5; x++) {
            D[x] = ROTL64(C[(x + 4) % 5], 1) ^ C[(x + 1) % 5];
        }
        for(int x = 0; x < 5; x++) {
            for(int y = 0; y < 25; y += 5) {
                state[x + y] ^= D[x];
            }
        }
        
        // Rho and Pi steps
        ulong B[25];
        for(int x = 0; x < 5; x++) {
            for(int y = 0; y < 5; y++) {
                B[y + ((2 * x + 3 * y) % 5) * 5] = ROTL64(state[x + y * 5], keccak_rotc[r]);
            }
        }
        
        // Chi step
        for(int y = 0; y < 5; y++) {
            for(int x = 0; x < 5; x++) {
                state[x + y * 5] = B[x + y * 5] ^ ((~B[((x + 1) % 5) + y * 5]) & B[((x + 2) % 5) + y * 5]);
            }
        }
        
        // Iota step
        state[0] ^= (1UL << (r));
    }
}

// Keccak256 hashing function
void keccak256(const uchar *input, size_t len, uchar *output) {
    // Initialize state to zero
    ulong state[25];
    for(int i = 0; i < 25; i++) {
        state[i] = 0;
    }
    
    // Absorb input
    size_t rate = 1088 / 8; // 1088 bits = 136 bytes
    size_t block_size = rate;
    size_t num_blocks = (len + block_size - 1) / block_size;
    
    for(size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        size_t current_block_size = (offset + block_size <= len) ? block_size : (len - offset);
        
        // XOR input block into state
        for(int j = 0; j < current_block_size; j++) {
            int byte_index = j;
            int lane = j / 8;
            int byte_in_lane = j % 8;
            state[lane] ^= ((ulong)input[offset + j]) << (8 * byte_in_lane);
        }
        
        // Apply permutation
        keccakf(state, KECCAK_ROUNDS);
    }
    
    // Padding
    // Append 0x01 to the end of the input
    size_t pad_offset = len;
    size_t pad_size = block_size - (len % block_size);
    uchar padded_block[136]; // block_size = 136 bytes
    for(int i = 0; i < 136; i++) {
        padded_block[i] = 0;
    }
    for(int i = 0; i < (len % block_size); i++) {
        padded_block[i] = input[pad_offset - (len % block_size) + i];
    }
    padded_block[len % block_size] = 0x01;
    padded_block[block_size - 1] |= 0x80;
    
    // XOR padded block into state
    for(int j = 0; j < block_size; j++) {
        int byte_index = j;
        int lane = j / 8;
        int byte_in_lane = j % 8;
        state[lane] ^= ((ulong)padded_block[j]) << (8 * byte_in_lane);
    }
    
    // Apply permutation
    keccakf(state, KECCAK_ROUNDS);
    
    // Squeeze output
    // Only need the first 256 bits (32 bytes)
    for(int i = 0; i < 32; i++) {
        int lane = i / 8;
        int byte_in_lane = i % 8;
        output[i] = (uchar)((state[lane] >> (8 * byte_in_lane)) & 0xFF);
    }
    
    // No dynamic memory allocation needed
}

// The main kernel function for mining
__kernel void hashMessage(
    __global uchar *message,          // 20-byte wallet address
    __global uchar *target,           // 32-byte target value
    ulong startPosition,              // Starting nonce position
    uint maxSolutionCount,            // Maximum number of solutions to store
    __global ulong *solutions,        // Buffer to store valid nonces
    __global uint *solution_count      // Buffer to store the count of valid solutions
) {
    // Define local storage for the target to optimize access speed
    __local uchar target_local[32];
    
    // Load the target into local memory
    for(int i = get_local_id(0); i < 32; i += get_local_size(0)) {
        target_local[i] = target[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure all work-items have loaded the target before proceeding
    
    // Initialize nonce vectors based on the global ID
    nonce_vec_t nonce_vec;
    nonce_vec.n0 = startPosition + get_global_id(0) * 4;
    nonce_vec.n1 = nonce_vec.n0 + 1;
    nonce_vec.n2 = nonce_vec.n0 + 2;
    nonce_vec.n3 = nonce_vec.n0 + 3;
    
    // Create a vector of nonces to process
    ulong4 nonces = (ulong4)(nonce_vec.n0, nonce_vec.n1, nonce_vec.n2, nonce_vec.n3);
    
    // Iterate over each nonce in the vector
    for(int i = 0; i < 4; i++) {
        ulong current_nonce = nonces[i];
        
        // Construct the full message by appending the nonce in big-endian format
        uchar full_message[52];
        // Copy the 20-byte wallet address
        for(int j = 0; j < 20; j++) {
            full_message[j] = message[j];
        }
        // Append the 32-byte nonce (big-endian)
        uchar nonce_bytes[32] = {0};
        for(int j = 0; j < 8; j++) { // 8 bytes for ulong
            nonce_bytes[31 - j] = (current_nonce >> (j * 8)) & 0xFF;
        }
        for(int j = 0; j < 32; j++) {
            full_message[20 + j] = nonce_bytes[j];
        }
        
        // Compute Keccak256 hash of the full message
        uchar hash[32];
        keccak256(full_message, 52, hash);
        
        // Compare the computed hash with the target
       bool is_valid = true;
        for(int k = 0; k < 32 && is_valid; k++) {
            if(hash[k] > target_local[k]) {
                is_valid = false;
            }
            else if(hash[k] < target_local[k]) {
                 break; // Early exit if hash is less than target at any point
                }
            }
        
        // If a valid nonce is found, store it atomically
        if(is_valid) {
            uint count = atomic_inc(solution_count);
            if(count < maxSolutionCount) {
                solutions[count] = current_nonce;
            }
        }
    }
}
