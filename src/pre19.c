// Preimage search for 19-step SHA-256 (standard IV, feed-forward), CPU only.
//
//   mode "comp":   target = 19-step compression-function output; message words free.
//   mode "padded": the message block must be a correctly padded 55-byte message
//                  (W13 & 0xff = 0x80, W14 = 0, W15 = 440), i.e. a real SHA-256 preimage
//                  of the 19-step hash function.
//
// Method (a-sequence + conditioned context + 3-level chain):
//   a_{-7..0} fixed by IV, a_{12..19} fixed by the target.  Context words a_5..a_11 are
//   chosen so that e_9 = e_10 = 0xffffffff.  Then, for early words a_1..a_4,
//     W_9  = -a_2 + c9  + R(a_3,a_4),   R = Maj(a_5,a_4,a_3) - a_4   (zero w.p. (3/4)^32)
//     W_10 = -a_3 + c10 + (a_4 & D),    D = a_5 ^ a_6
//     W_11 = -a_4 + c11
//   and the schedule constraints C16..C18 become
//     Q(W_1) = T16 + G1(a_1)       - W_0 - R      (Q(u) = sigma0(u) - u)
//     Q(W_2) = T17 + G2(a_1,a_2)   - W_1 - (a_4&D)
//     Q(W_3) = T18 + G3(a_1..a_3)  - W_2
//   solved in sequence with a partial inverse table of Q; success needs R = 0, a_4&D = 0.
//
// usage: pre19 comp|padded <target_hex_64> <table_bits> <seed> [kmax] [chains_per_context]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint32_t u32;
typedef uint64_t u64;

static const u32 K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static u32 IV[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
static u64 PADLEN_BITS = 440;   /* total message length in bits (last block holds 55 bytes) */

static inline u32 ROR(u32 x, int n){return (x>>(n&31))|(x<<((-n)&31));}
static inline u32 S0(u32 x){return ROR(x,2)^ROR(x,13)^ROR(x,22);}
static inline u32 S1(u32 x){return ROR(x,6)^ROR(x,11)^ROR(x,25);}
static inline u32 s0(u32 x){return ROR(x,7)^ROR(x,18)^(x>>3);}
static inline u32 s1(u32 x){return ROR(x,17)^ROR(x,19)^(x>>10);}
static inline u32 CH(u32 e,u32 f,u32 g){return (e&f)^(~e&g);}
static inline u32 MAJ(u32 a,u32 b,u32 c){return (a&b)^(a&c)^(b&c);}
static inline u32 Qf(u32 u){return s0(u)-u;}

#define R 19
#define OFF 7
static u32 A[R + OFF + 2];                 // A[t+OFF] = a_t, t in [-7, R]
static inline u32 E(int s){ return A[s+OFF] - S0(A[s-1+OFF]) - MAJ(A[s-1+OFF],A[s-2+OFF],A[s-3+OFF]) + A[s-4+OFF]; }
static inline u32 Wt(int t){ u32 et=E(t); return E(t+1) - A[t-3+OFF] - E(t-3) - S1(et) - CH(et,E(t-1),E(t-2)) - K[t]; }

// ---------- RNG ----------
static u64 rs;
static inline u64 rnd64(void){ u64 z=(rs+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);}
static inline u32 rnd32(void){ return (u32)rnd64(); }

// ---------- partial inverse table of Q ----------
#define TB 24
static u32 *tabU, *offs;
static void build_table(int m){
    u64 n = 1ULL<<m;
    offs = calloc((1u<<TB)+2, sizeof(u32));
    tabU = malloc(n*sizeof(u32));
    if(!offs||!tabU){fprintf(stderr,"alloc failed\n"); exit(1);}
    for(u64 i=0;i<n;i++){ u32 u=(u32)(i*0x9E3779B1u); offs[(Qf(u)>>(32-TB))+1]++; }
    for(u64 b=1;b<=(1u<<TB);b++) offs[b]+=offs[b-1];
    u32 *pos = malloc(((1u<<TB)+1)*sizeof(u32)); memcpy(pos,offs,((1u<<TB)+1)*sizeof(u32));
    for(u64 i=0;i<n;i++){ u32 u=(u32)(i*0x9E3779B1u); tabU[pos[Qf(u)>>(32-TB)]++]=u; }
    free(pos);
}
static inline int lookup(u32 y, u32 *out){
    u32 b=y>>(32-TB); int k=0;
    for(u32 i=offs[b];i<offs[b+1];i++){ u32 u=tabU[i]; if(Qf(u)==y && k<8) out[k++]=u; }
    return k;
}

// ---------- helpers ----------
static void window_from_state(const u32 st[8], u32 w[8]){   // w[0..7] = a_{r-7..r}
    u32 Aa[8], Ee[4];
    for(int i=0;i<4;i++){ Aa[i]=st[i]; Ee[i]=st[4+i]; }
    for(int k=0;k<4;k++) Aa[4+k] = Ee[k] - Aa[k] + S0(Aa[k+1]) + MAJ(Aa[k+1],Aa[k+2],Aa[k+3]);
    for(int i=0;i<8;i++) w[i]=Aa[7-i];
}
static void compress(const u32 Wm[16], int r, u32 out[8]){
    u32 W[64]; memcpy(W,Wm,64);
    for(int t=16;t<r;t++) W[t]=s1(W[t-2])+W[t-7]+s0(W[t-15])+W[t-16];
    u32 a=IV[0],b=IV[1],c=IV[2],d=IV[3],e=IV[4],f=IV[5],g=IV[6],h=IV[7];
    for(int t=0;t<r;t++){ u32 T1=h+S1(e)+CH(e,f,g)+K[t]+W[t], T2=S0(a)+MAJ(a,b,c);
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2; }
    u32 st[8]={a,b,c,d,e,f,g,h}; for(int i=0;i<8;i++) out[i]=st[i]+IV[i];
}
static int popc(u32 x){ return __builtin_popcount(x); }

int main(int argc, char **argv){
    if(argc<5){ fprintf(stderr,"usage: %s comp|padded target_hex table_bits seed [kmax] [log2chains]\n",argv[0]); return 1; }
    int padded = strcmp(argv[1],"padded")==0;
    if(!padded && strcmp(argv[1],"comp")!=0){
        fprintf(stderr,"error: mode must be 'comp' or 'padded'\n");
        return 1;
    }
    int m = atoi(argv[3]);
    if(m < 1 || m > 31){
        fprintf(stderr,"error: table_bits must be between 1 and 31 (got %d)\n", m);
        return 1;
    }
    size_t targ_len = strlen(argv[2]);
    if(targ_len < 64 || ((targ_len + 1) % 65) != 0){
        fprintf(stderr,"error: target_hex must be one or more 64-character hex digests separated by single delimiters\n");
        return 1;
    }
    int ntarg = (int)((targ_len+1)/65);
    for(int ti=0; ti<ntarg; ti++){
        for(int j=0; j<64; j++){
            char c = argv[2][65*ti+j];
            if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F'))){
                fprintf(stderr,"error: invalid hex character '%c' in target_hex at offset %d\n", c, 65*ti+j);
                return 1;
            }
        }
    }
    rs = strtoull(argv[4],0,10)*0x2545F4914F6CDD1DULL + 7;
    int kmax = argc>5 ? atoi(argv[5]) : 3;
    if(kmax < 1 || kmax > 8){
        fprintf(stderr,"error: kmax must be between 1 and 8 (got %d)\n", kmax);
        return 1;
    }
    int lgch = argc>6 ? atoi(argv[6]) : 26;
    if(lgch < 1 || lgch > 32){
        fprintf(stderr,"error: log2chains must be between 1 and 32 (got %d)\n", lgch);
        return 1;
    }
    for(int ai=7; ai<argc; ai++){
        if(!strncmp(argv[ai],"iv=",3)){
            const char *iv_hex = argv[ai] + 3;
            if(strlen(iv_hex) != 64){
                fprintf(stderr,"error: iv= must be followed by exactly 64 hex characters\n");
                return 1;
            }
            for(int j=0; j<64; j++){
                char c = iv_hex[j];
                if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F'))){
                    fprintf(stderr,"error: invalid hex character '%c' in iv= at offset %d\n", c, j);
                    return 1;
                }
            }
            for(int i=0;i<8;i++){ char b[9]; memcpy(b,iv_hex+8*i,8); b[8]=0; IV[i]=(u32)strtoul(b,0,16);}
        }
        else if(!strncmp(argv[ai],"len=",4)){
            const char *len_str = argv[ai] + 4;
            char *endp = NULL;
            if(*len_str == '\0'){
                fprintf(stderr,"error: len= requires a non-empty integer byte length\n");
                return 1;
            }
            unsigned long long len_bytes = strtoull(len_str, &endp, 10);
            if(!endp || *endp != '\0'){
                fprintf(stderr,"error: invalid integer in len=%s\n", len_str);
                return 1;
            }
            PADLEN_BITS = 8ULL * len_bytes;
        } else {
            fprintf(stderr,"error: unknown optional argument '%s' (expected iv=<64hex> or len=<bytes>)\n", argv[ai]);
            return 1;
        }
    }
    u64 chains_per_ctx = 1ULL<<lgch;

    clock_t t0=clock();
    build_table(m);
    fprintf(stderr,"table built: 2^%d entries (%.1f s)\n", m, (double)(clock()-t0)/CLOCKS_PER_SEC);

  for(int ti=0; ti<ntarg; ti++){
    u32 dig[8];
    for(int i=0;i<8;i++){ char buf[9]; memcpy(buf,argv[2]+65*ti+8*i,8); buf[8]=0; dig[i]=(u32)strtoul(buf,0,16); }
    clock_t tt=clock();
    // IV part of the a-sequence
    u32 wiv[8]; window_from_state(IV, wiv);
    for(int i=0;i<8;i++) A[i]=wiv[i];            // a_{-7..0}
    // target part
    u32 st[8]; for(int i=0;i<8;i++) st[i]=dig[i]-IV[i];
    u32 wt[8]; window_from_state(st, wt);
    for(int i=0;i<8;i++) A[(R-7+i)+OFF]=wt[i];    // a_{12..19}

    u64 ctx_tried=0, ctx_used=0, chains=0;
    u32 cand1[8],cand2[8],cand3[8];
    int solved=0;
    while(!solved){
        // ---------- context ----------
        u32 D;
        if(!padded){
            u32 v=rnd32(); A[5+OFF]=v; A[6+OFF]=v;
            A[7+OFF]=rnd32(); A[8+OFF]=rnd32(); A[11+OFF]=rnd32();
            A[9+OFF]  = 0xffffffffu + S0(A[8+OFF]) + MAJ(A[8+OFF],A[7+OFF],A[6+OFF]) - A[5+OFF];
            A[10+OFF] = 0xffffffffu + S0(A[9+OFF]) + MAJ(A[9+OFF],A[8+OFF],A[7+OFF]) - A[6+OFF];
            D = 0; ctx_tried++;
        } else {
            for(;;){
                ctx_tried++;
                A[9+OFF]=rnd32(); A[10+OFF]=rnd32(); A[11+OFF]=rnd32();
                A[8+OFF]=0; u32 w15=Wt(15); A[8+OFF]=w15-(u32)PADLEN_BITS;    // W15 = low word of bit length
                A[7+OFF]=0; u32 w14=Wt(14); A[7+OFF]=w14-(u32)(PADLEN_BITS>>32); // W14 = high word of bit length
                A[6+OFF] = 0xffffffffu - A[10+OFF] + S0(A[9+OFF]) + MAJ(A[9+OFF],A[8+OFF],A[7+OFF]);  // e10=-1
                A[5+OFF]=0; A[4+OFF]=0; A[3+OFF]=0; A[2+OFF]=0;
                if((Wt(13)&0xff)!=0x80) continue;                          // W13 low byte (independent of a5, a6)
                A[5+OFF] = 0xffffffffu - A[9+OFF] + S0(A[8+OFF]) + MAJ(A[8+OFF],A[7+OFF],A[6+OFF]);  // e9=-1
                D = A[5+OFF]^A[6+OFF];
                if(popc(D)<=kmax) break;
            }
        }
        if(E(9)!=0xffffffffu || E(10)!=0xffffffffu){ fprintf(stderr,"condition bug\n"); return 2; }
        ctx_used++;
        // ---------- constants ----------
        A[1+OFF]=A[2+OFF]=A[3+OFF]=A[4+OFF]=0;
        u32 c9=Wt(9), c10=Wt(10), c11=Wt(11);
        u32 W14=Wt(14),W15=Wt(15),W16=Wt(16),W17=Wt(17),W18=Wt(18);
        u32 T16=W16-s1(W14)-c9, T17=W17-s1(W15)-c10, T18=W18-s1(W16)-c11;
        // ---------- chain ----------
        u32 base=rnd32();
        for(u64 i=0;i<chains_per_ctx;i++){
            chains++;
            u32 a1=base+(u32)i;
            A[1+OFF]=a1; A[2+OFF]=0;
            u32 W0=Wt(0), G1=0u-Wt(1);
            u32 t16=T16+G1-W0;                 // R guessed 0
            int n1=lookup(t16,cand1);
            for(int x=0;x<n1;x++){
                u32 W1=cand1[x], a2=W1+G1;
                A[2+OFF]=a2; A[3+OFF]=0;
                u32 G2=0u-Wt(2);
                u32 t17=T17+G2-W1;             // (a4 & D) guessed 0
                int n2=lookup(t17,cand2);
                for(int y=0;y<n2;y++){
                    u32 W2=cand2[y], a3=W2+G2;
                    A[3+OFF]=a3; A[4+OFF]=0;
                    u32 G3=0u-Wt(3);
                    u32 t18=T18+G3-W2;
                    int n3=lookup(t18,cand3);
                    for(int z=0;z<n3;z++){
                        u32 W3=cand3[z], a4=W3+G3;
                        if(((MAJ(A[5+OFF],a4,a3)-a4))!=0u) continue;
                        if((a4&D)!=0u) continue;
                        A[4+OFF]=a4;
                        u32 Wm[16]; for(int t=0;t<16;t++) Wm[t]=Wt(t);
                        u32 out[8]; compress(Wm,R,out);
                        int ok=1; for(int k=0;k<8;k++) ok&=(out[k]==dig[k]);
                        if(padded) ok &= ((Wm[13]&0xff)==0x80) && Wm[14]==(u32)(PADLEN_BITS>>32) && Wm[15]==(u32)PADLEN_BITS;
                        printf("FOUND ok=%d mode=%s target=%d contexts_tried=%llu contexts_used=%llu chains=%llu search_time=%.1fs\n",
                               ok, padded?"padded":"comp", ti, (unsigned long long)ctx_tried,(unsigned long long)ctx_used,
                               (unsigned long long)chains,(double)(clock()-tt)/CLOCKS_PER_SEC);
                        printf("W ="); for(int t=0;t<16;t++) printf(" %08x",Wm[t]); printf("\n");
                        printf("H19="); for(int k=0;k<8;k++) printf("%08x",out[k]); printf("\n");
                        fflush(stdout);
                        if(ok){ solved=1; goto next_ctx; }
                    }
                }
            }
        }
        fprintf(stderr,"context %llu exhausted (chains so far %llu, %.1fs)\n",(unsigned long long)ctx_used,(unsigned long long)chains,(double)(clock()-tt)/CLOCKS_PER_SEC);
        next_ctx: ;
    }
  }
  return 0;
}
