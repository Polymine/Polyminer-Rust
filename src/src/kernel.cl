
#define ADDRESS_LENGTH    20
#define UINT64_LENGTH     8
#define UINT256_LENGTH    32
#define MESSAGE_LENGTH    (ADDRESS_LENGTH + UINT256_LENGTH) // 52 bytes
#define NONCE_POSITION    ADDRESS_LENGTH
#define SPONGE_LENGTH     200

typedef union _nonce_t {
    ulong uint64_t;
    uchar uint8_t[UINT64_LENGTH];
} nonce_t;

static inline ulong rol64(ulong x, uint s) {
    return ((x << s) | (x >> (64u - s)));
}

__constant ulong Keccak_f1600_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x8000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant uchar rho[24] = {
    1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44
};

__constant uchar pi[24] = {
    10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
};

static inline void xorin(uchar* dst, const uchar* src, uint len) {
    for (uint i = 0; i < len; i++) {
        dst[i] ^= src[i];
    }
}

static inline void keccakf(uchar *state) {
    ulong *a = (ulong *)state;
    ulong b[5];
    ulong t;

    #pragma unroll
    for (uint i = 0; i < 24; i++) {
        b[0] = a[0]^a[5]^a[10]^a[15]^a[20];
        b[1] = a[1]^a[6]^a[11]^a[16]^a[21];
        b[2] = a[2]^a[7]^a[12]^a[17]^a[22];
        b[3] = a[3]^a[8]^a[13]^a[18]^a[23];
        b[4] = a[4]^a[9]^a[14]^a[19]^a[24];

        a[0]^=b[4]^rol64(b[1],1); a[5]^=b[4]^rol64(b[1],1); a[10]^=b[4]^rol64(b[1],1);
        a[15]^=b[4]^rol64(b[1],1); a[20]^=b[4]^rol64(b[1],1);

        a[1]^=b[0]^rol64(b[2],1); a[6]^=b[0]^rol64(b[2],1); a[11]^=b[0]^rol64(b[2],1);
        a[16]^=b[0]^rol64(b[2],1); a[21]^=b[0]^rol64(b[2],1);

        a[2]^=b[1]^rol64(b[3],1); a[7]^=b[1]^rol64(b[3],1); a[12]^=b[1]^rol64(b[3],1);
        a[17]^=b[1]^rol64(b[3],1); a[22]^=b[1]^rol64(b[3],1);

        a[3]^=b[2]^rol64(b[4],1); a[8]^=b[2]^rol64(b[4],1); a[13]^=b[2]^rol64(b[4],1);
        a[18]^=b[2]^rol64(b[4],1); a[23]^=b[2]^rol64(b[4],1);

        a[4]^=b[3]^rol64(b[0],1); a[9]^=b[3]^rol64(b[0],1); a[14]^=b[3]^rol64(b[0],1);
        a[19]^=b[3]^rol64(b[0],1); a[24]^=b[3]^rol64(b[0],1);

        t = a[1];
        for(uint x=0; x<24; x++){
            ulong tmp = a[pi[x]];
            a[pi[x]] = rol64(t, rho[x]);
            t = tmp;
        }

        for(uint y=0; y<25; y+=5){
            ulong c0 = a[y];
            ulong c1 = a[y+1];
            ulong c2 = a[y+2];
            ulong c3 = a[y+3];
            ulong c4 = a[y+4];
            a[y+0] = c0 ^ ((~c1) & c2);
            a[y+1] = c1 ^ ((~c2) & c3);
            a[y+2] = c2 ^ ((~c3) & c4);
            a[y+3] = c3 ^ ((~c4) & c0);
            a[y+4] = c4 ^ ((~c0) & c1);
        }
        a[0] ^= Keccak_f1600_RC[i];
    }
}

static inline void keccak256(uchar *digest, const uchar *message) {
    uchar sponge[SPONGE_LENGTH];
    #pragma unroll
    for(uint i=0; i<SPONGE_LENGTH; i++) sponge[i]=0;

    xorin(sponge, message, MESSAGE_LENGTH);
    sponge[MESSAGE_LENGTH] ^= 0x01u;
    sponge[SPONGE_LENGTH-1] ^= 0x80u;

    keccakf(sponge);

    #pragma unroll
    for(uint i=0; i<UINT256_LENGTH; i++)
        digest[i] = sponge[i];
}

static inline int compare(const uchar *left, const __local uchar *right, uint len) {
    for (uint i = 0; i < len; i++) {
        if (left[i] < right[i]) {
            return -1;
        } else if (left[i] > right[i]) {
            return 1;
        }
    }
    return 0;
}


__kernel void hashMessage(
    __global const uchar *d_message, 
    __global const uchar *d_target,
    ulong startPosition,
    uint maxSolutionCount,
    __global ulong *solutions,
    __global uint *solutionCount)
{
    uchar digest[UINT256_LENGTH];
    uchar message[MESSAGE_LENGTH];
    __local uchar target_local[UINT256_LENGTH];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        for(uint i = 0; i < UINT256_LENGTH; i++) {
            target_local[i] = d_target[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i=0; i<MESSAGE_LENGTH; i++)
        message[i] = d_message[i];

    nonce_t nonce;
    nonce.uint64_t = startPosition + get_global_id(0);

    for (uint i=0; i<UINT256_LENGTH; i++) {
        if (i < 24u) {
            message[NONCE_POSITION + i] = 0u;
        } else {
            message[NONCE_POSITION + i] = nonce.uint8_t[UINT64_LENGTH - 1 - (i - 24)];
        }
    }

    keccak256(digest, message);

    if (compare(digest, (__local uchar *)target_local, UINT256_LENGTH) < 0) {
        uint idx = atomic_inc(&solutionCount[0]);
        if(idx < maxSolutionCount){
            solutions[idx] = nonce.uint64_t;
        }
    }
}
