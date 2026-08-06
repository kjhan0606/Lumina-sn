/* lumina_atomic.c — Phase 2 - Step 7: Load TARDIS reference data from CSV/NPY files.
 * Reads the exact converged plasma state exported by export_tardis_reference.py.
 * This ensures bit-for-bit matching with TARDIS ground truth. */

#include "lumina.h" /* Phase 2 - Step 7 */
#include "seed_capability.h"
#include <errno.h>  /* composition parser: distinguish ERANGE from valid zero */
#include <limits.h> /* composition parser: validate Z before narrowing to int */

#ifdef __cplusplus   /* Phase 6 - Step 9: extern C guard for NVCC */
extern "C" {         /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

/* ============================================================ */
/* Phase 2 - Step 8: NPY file reader (NumPy .npy format)       */
/* ============================================================ */

/* K-SHAPE uses SHA-256 to bind both runtime arrays to the exact line_list.csv
 * epoch.  Keep this implementation local: several CPU fixtures link atomic.c
 * without lumina_cmfgen.c, so depending on its hash helper would break them. */
typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} KShapeSHA256;

static uint32_t kshape_rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32U - n));
}

static void kshape_sha256_transform(KShapeSHA256 *s,
                                    const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,
        0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,
        0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,
        0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,
        0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,
        0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,
        0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,
        0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,
        0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[4*i] << 24) |
               ((uint32_t)block[4*i+1] << 16) |
               ((uint32_t)block[4*i+2] << 8) | block[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t a = w[i-15], b = w[i-2];
        uint32_t s0 = kshape_rotr32(a,7) ^ kshape_rotr32(a,18) ^ (a >> 3);
        uint32_t s1 = kshape_rotr32(b,17) ^ kshape_rotr32(b,19) ^ (b >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3];
    uint32_t e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1=kshape_rotr32(e,6)^kshape_rotr32(e,11)^kshape_rotr32(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+s1+ch+k[i]+w[i];
        uint32_t s0=kshape_rotr32(a,2)^kshape_rotr32(a,13)^kshape_rotr32(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=s0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;
    s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}

static void kshape_sha256_init(KShapeSHA256 *s) {
    static const uint32_t init[8] = {
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U
    };
    memcpy(s->h, init, sizeof(init)); s->bits=0; s->used=0;
}

static void kshape_sha256_update(KShapeSHA256 *s, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    s->bits += (uint64_t)n * 8U;
    while (n) {
        size_t take = 64U - s->used;
        if (take > n) take = n;
        memcpy(s->block+s->used,p,take); s->used+=take; p+=take; n-=take;
        if (s->used == 64U) { kshape_sha256_transform(s,s->block); s->used=0; }
    }
}

static void kshape_sha256_final(KShapeSHA256 *s, unsigned char out[32]) {
    uint64_t bits=s->bits;
    unsigned char one=0x80,zero=0,len[8];
    for(int i=0;i<8;i++) len[7-i]=(unsigned char)(bits>>(8*i));
    kshape_sha256_update(s,&one,1);
    while(s->used!=56) kshape_sha256_update(s,&zero,1);
    kshape_sha256_update(s,len,8);
    for(int i=0;i<8;i++) {
        out[4*i]=(unsigned char)(s->h[i]>>24);
        out[4*i+1]=(unsigned char)(s->h[i]>>16);
        out[4*i+2]=(unsigned char)(s->h[i]>>8);
        out[4*i+3]=(unsigned char)s->h[i];
    }
}

static int kshape_sha256_file(const char *path, char hex[65]) {
    FILE *fp=fopen(path,"rb");
    if(!fp) return -1;
    KShapeSHA256 sha; unsigned char buf[65536],digest[32];
    kshape_sha256_init(&sha);
    for (;;) {
        size_t n=fread(buf,1,sizeof(buf),fp);
        if(n) kshape_sha256_update(&sha,buf,n);
        if(n<sizeof(buf)) { if(ferror(fp)){fclose(fp);return -1;} break; }
    }
    if(fclose(fp)) return -1;
    kshape_sha256_final(&sha,digest);
    for(int i=0;i<32;i++) snprintf(hex+2*i,3,"%02x",digest[i]);
    hex[64]='\0'; return 0;
}

/* Strict K array reader: exactly 2-D, C-order, little-endian float64, and no
 * truncated or trailing payload.  The legacy generic reader remains for zeta
 * and integer-conversion companions with their historical formats. */
static double *read_npy_f64_strict_2d(const char *path,
                                      int *out_rows, int *out_cols) {
    FILE *fp=NULL; char *header=NULL; double *data=NULL;
    *out_rows=0; *out_cols=0;
    fp=fopen(path,"rb");
    if(!fp) { fprintf(stderr,"[K-SHAPE][FATAL] cannot open %s: %s\n",path,strerror(errno)); return NULL; }
    unsigned char preamble[12];
    if(fread(preamble,1,8,fp)!=8 || memcmp(preamble,"\x93NUMPY",6)!=0) {
        fprintf(stderr,"[K-SHAPE][FATAL] invalid/truncated NPY preamble: %s\n",path); goto fail;
    }
    uint32_t hlen=0;
    if(preamble[6]==1) {
        if(fread(preamble+8,1,2,fp)!=2) { fprintf(stderr,"[K-SHAPE][FATAL] truncated NPY v1 header length: %s\n",path); goto fail; }
        hlen=(uint32_t)preamble[8]|((uint32_t)preamble[9]<<8);
    } else if(preamble[6]==2 || preamble[6]==3) {
        if(fread(preamble+8,1,4,fp)!=4) { fprintf(stderr,"[K-SHAPE][FATAL] truncated NPY header length: %s\n",path); goto fail; }
        hlen=(uint32_t)preamble[8]|((uint32_t)preamble[9]<<8)|
             ((uint32_t)preamble[10]<<16)|((uint32_t)preamble[11]<<24);
    } else {
        fprintf(stderr,"[K-SHAPE][FATAL] unsupported NPY version %u.%u: %s\n",
                preamble[6],preamble[7],path); goto fail;
    }
    if(hlen==0 || hlen>(1U<<20)) { fprintf(stderr,"[K-SHAPE][FATAL] invalid NPY header length %u: %s\n",hlen,path); goto fail; }
    header=(char*)malloc((size_t)hlen+1U);
    if(!header || fread(header,1,hlen,fp)!=(size_t)hlen) {
        fprintf(stderr,"[K-SHAPE][FATAL] truncated NPY header: %s\n",path); goto fail;
    }
    header[hlen]='\0';
    if(!(strstr(header,"'descr': '<f8'") || strstr(header,"\"descr\": \"<f8\""))) {
        fprintf(stderr,"[K-SHAPE][FATAL] %s dtype/byte-order must be little-endian float64 (<f8)\n",path); goto fail;
    }
    if(!(strstr(header,"'fortran_order': False") || strstr(header,"\"fortran_order\": false"))) {
        fprintf(stderr,"[K-SHAPE][FATAL] %s must be C-order (fortran_order=False)\n",path); goto fail;
    }
    char *shape=strstr(header,"'shape': (");
    if(!shape) shape=strstr(header,"\"shape\": (");
    if(!shape || !(shape=strchr(shape,'('))) {
        fprintf(stderr,"[K-SHAPE][FATAL] missing NPY shape: %s\n",path); goto fail;
    }
    char *end=NULL; errno=0;
    unsigned long long rows=strtoull(shape+1,&end,10);
    if(errno || end==shape+1 || !end) { fprintf(stderr,"[K-SHAPE][FATAL] invalid NPY row count: %s\n",path); goto fail; }
    while(*end==' ') end++;
    if(*end!=',') { fprintf(stderr,"[K-SHAPE][FATAL] %s must be exactly 2-D\n",path); goto fail; }
    char *colstart=end+1; while(*colstart==' ') colstart++;
    errno=0; unsigned long long cols=strtoull(colstart,&end,10);
    if(errno || end==colstart || !end) { fprintf(stderr,"[K-SHAPE][FATAL] invalid NPY column count: %s\n",path); goto fail; }
    while(*end==' ' || *end==',') end++;
    if(*end!=')' || rows==0 || cols==0 || rows>INT_MAX || cols>INT_MAX ||
       rows>SIZE_MAX/cols || rows*cols>SIZE_MAX/sizeof(double)) {
        fprintf(stderr,"[K-SHAPE][FATAL] invalid/overflowing NPY shape (%llu,%llu): %s\n",rows,cols,path); goto fail;
    }
    { const uint16_t endian_probe=1;
      if(*(const unsigned char*)&endian_probe!=1) {
          fprintf(stderr,"[K-SHAPE][FATAL] little-endian NPY on non-little-endian host: %s\n",path); goto fail;
      } }
    size_t total=(size_t)rows*(size_t)cols;
    data=(double*)malloc(total*sizeof(double));
    if(!data || fread(data,sizeof(double),total,fp)!=total) {
        fprintf(stderr,"[K-SHAPE][FATAL] truncated NPY payload: %s\n",path); goto fail;
    }
    if(fgetc(fp)!=EOF || ferror(fp)) {
        fprintf(stderr,"[K-SHAPE][FATAL] trailing/corrupt NPY payload: %s\n",path); goto fail;
    }
    free(header); fclose(fp);
    *out_rows=(int)rows; *out_cols=(int)cols; return data;
fail:
    free(header); free(data); fclose(fp); return NULL;
}

typedef struct {
    char schema[32], line_hash[65], tau_hash[65], trans_hash[65];
    int n_lines, n_trans, n_shells;
    int seen_schema, seen_line, seen_tau, seen_trans, seen_nlines, seen_ntrans,
        seen_nshells, seen_dtype, seen_byte_order, seen_order;
} KShapeContract;

static int validate_kshape_contract(const char *ref_dir, int n_lines,
                                    int n_trans, int n_shells) {
    char path[512],line_path[512],tau_path[512],trans_path[512];
    snprintf(path,sizeof(path),"%s/kshape_contract.txt",ref_dir);
    FILE *fp=fopen(path,"r");
    if(!fp) { fprintf(stderr,"[K-SHAPE][FATAL] missing contract %s: %s\n",path,strerror(errno)); return -1; }
    KShapeContract c; memset(&c,0,sizeof(c));
    char line[256]; int lineno=0,failed=0;
    while(fgets(line,sizeof(line),fp)) {
        lineno++; size_t len=strlen(line);
        if(len==0 || line[len-1]!='\n') { fprintf(stderr,"[K-SHAPE][FATAL] malformed/overlong %s:%d\n",path,lineno); failed=1; break; }
        line[--len]='\0'; if(len && line[len-1]=='\r') line[--len]='\0';
        char *eq=strchr(line,'='); if(!eq || eq==line || !eq[1]) { fprintf(stderr,"[K-SHAPE][FATAL] malformed %s:%d\n",path,lineno); failed=1; break; }
        *eq='\0'; const char *key=line,*value=eq+1;
#define KSTR(k,field,seen) if(!strcmp(key,k)){ if(c.seen){failed=1;break;} c.seen=1; snprintf(c.field,sizeof(c.field),"%s",value); }
#define KINT(k,field,seen) if(!strcmp(key,k)){ char *e=NULL; long v=strtol(value,&e,10); if(c.seen||!e||*e||v<1||v>INT_MAX){failed=1;break;} c.seen=1;c.field=(int)v; }
        if(!strcmp(key,"schema")){ if(c.seen_schema){failed=1;break;} c.seen_schema=1;snprintf(c.schema,sizeof(c.schema),"%s",value); }
        else KSTR("line_list_sha256",line_hash,seen_line)
        else KSTR("tau_sobolev_sha256",tau_hash,seen_tau)
        else KSTR("transition_probabilities_sha256",trans_hash,seen_trans)
        else KINT("n_lines",n_lines,seen_nlines)
        else KINT("n_macro_transitions",n_trans,seen_ntrans)
        else KINT("n_shells",n_shells,seen_nshells)
        else if(!strcmp(key,"dtype")){ if(c.seen_dtype||strcmp(value,"<f8")){failed=1;break;} c.seen_dtype=1; }
        else if(!strcmp(key,"byte_order")){ if(c.seen_byte_order||strcmp(value,"little")){failed=1;break;} c.seen_byte_order=1; }
        else if(!strcmp(key,"array_order")){ if(c.seen_order||strcmp(value,"C")){failed=1;break;} c.seen_order=1; }
        else { failed=1; break; }
#undef KSTR
#undef KINT
    }
    if(ferror(fp)) failed=1;
    fclose(fp);
    int all_seen=c.seen_schema+c.seen_line+c.seen_tau+c.seen_trans+
        c.seen_nlines+c.seen_ntrans+c.seen_nshells+c.seen_dtype+
        c.seen_byte_order+c.seen_order;
    if(failed || all_seen!=10 || strcmp(c.schema,"lumina-kshape-v1") ||
       c.n_lines!=n_lines || c.n_trans!=n_trans || c.n_shells!=n_shells) {
        fprintf(stderr,"[K-SHAPE][FATAL] invalid contract %s (lines=%d/%d transitions=%d/%d shells=%d/%d)\n",
                path,c.n_lines,n_lines,c.n_trans,n_trans,c.n_shells,n_shells); return -1;
    }
    snprintf(line_path,sizeof(line_path),"%s/line_list.csv",ref_dir);
    snprintf(tau_path,sizeof(tau_path),"%s/tau_sobolev.npy",ref_dir);
    snprintf(trans_path,sizeof(trans_path),"%s/transition_probabilities.npy",ref_dir);
    char actual[65];
#define KHASH(file,expected,label) do { if(kshape_sha256_file(file,actual)!=0 || strcmp(actual,expected)){ fprintf(stderr,"[K-SHAPE][FATAL] %s hash/line-epoch mismatch: %s\n",label,file); return -1; } } while(0)
    KHASH(line_path,c.line_hash,"line_list");
    KHASH(tau_path,c.tau_hash,"tau_sobolev");
    KHASH(trans_path,c.trans_hash,"transition_probabilities");
#undef KHASH
    printf("  K-SHAPE contract: line epoch %.16s...; dtype=<f8 byte_order=little order=C\n",c.line_hash);
    return 0;
}

/* Phase 2 - Step 8: Read NPY header, return data pointer */
static double *read_npy_f64(const char *path, int *out_rows, int *out_cols) {
    FILE *fp = fopen(path, "rb"); /* Phase 2 - Step 8 */
    if (!fp) { /* Phase 2 - Step 8 */
        fprintf(stderr, "ERROR: Cannot open %s\n", path); /* Phase 2 - Step 8 */
        return NULL; /* Phase 2 - Step 8 */
    }

    /* A7: short-read aware macro — abort cleanly instead of trusting garbage. */
    #define NPY_FREAD(buf, elt, cnt) \
        do { \
            size_t _need = (size_t)(cnt); \
            size_t _got  = fread((buf), (elt), _need, fp); \
            if (_got != _need) { \
                fprintf(stderr, "ERROR: %s short read (%zu/%zu) at line %d\n", \
                        path, _got, _need, __LINE__); \
                fclose(fp); \
                return NULL; \
            } \
        } while (0)

    /* Phase 2 - Step 8: Read magic number */
    unsigned char magic[6]; /* Phase 2 - Step 8 */
    NPY_FREAD(magic, 1, 6); /* A7 */
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || /* Phase 2 - Step 8 */
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') { /* Phase 2 - Step 8 */
        fprintf(stderr, "ERROR: %s is not a valid .npy file\n", path); /* Phase 2 - Step 8 */
        fclose(fp); /* Phase 2 - Step 8 */
        return NULL; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Read version */
    unsigned char version[2]; /* Phase 2 - Step 8 */
    NPY_FREAD(version, 1, 2); /* A7 */

    /* Phase 2 - Step 8: Read header length */
    uint16_t header_len; /* Phase 2 - Step 8 */
    if (version[0] == 1) { /* Phase 2 - Step 8 */
        NPY_FREAD(&header_len, 2, 1); /* A7 */
    } else { /* Phase 2 - Step 8 */
        uint32_t hl32; /* Phase 2 - Step 8 */
        NPY_FREAD(&hl32, 4, 1); /* A7 */
        header_len = (uint16_t)hl32; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Read header string */
    char *header = (char *)malloc(header_len + 1); /* Phase 2 - Step 8 */
    {
        size_t _got = fread(header, 1, header_len, fp);
        if (_got != (size_t)header_len) {
            fprintf(stderr, "ERROR: %s short read of header (%zu/%u)\n",
                    path, _got, (unsigned)header_len);
            free(header); fclose(fp); return NULL;
        }
    }
    header[header_len] = '\0'; /* Phase 2 - Step 8 */

    /* Phase 2 - Step 8: Parse shape from header */
    int rows = 0, cols = 0; /* Phase 2 - Step 8 */
    char *shape_start = strstr(header, "'shape': ("); /* Phase 2 - Step 8 */
    if (!shape_start) shape_start = strstr(header, "\"shape\": ("); /* Phase 2 - Step 8 */
    if (shape_start) { /* Phase 2 - Step 8 */
        shape_start = strchr(shape_start, '(') + 1; /* Phase 2 - Step 8 */
        char *shape_end = strchr(shape_start, ')'); /* Phase 2 - Step 8 */
        char shape_str[256]; /* Phase 2 - Step 8 */
        int len = (int)(shape_end - shape_start); /* Phase 2 - Step 8 */
        strncpy(shape_str, shape_start, len); /* Phase 2 - Step 8 */
        shape_str[len] = '\0'; /* Phase 2 - Step 8 */

        /* Phase 2 - Step 8: Count commas for dimensionality */
        rows = atoi(shape_str); /* Phase 2 - Step 8 */
        char *comma = strchr(shape_str, ','); /* Phase 2 - Step 8 */
        if (comma) { /* Phase 2 - Step 8 */
            /* Phase 2 - Step 8: Skip whitespace after comma */
            char *after = comma + 1; /* Phase 2 - Step 8 */
            while (*after == ' ') after++; /* Phase 2 - Step 8 */
            if (*after != '\0' && *after != ')') { /* Phase 2 - Step 8: 2D */
                cols = atoi(after); /* Phase 2 - Step 8 */
                if (cols <= 0) cols = 1; /* Phase 2 - Step 8 */
            } else { /* Phase 2 - Step 8: 1D with trailing comma */
                cols = 1; /* Phase 2 - Step 8 */
            }
        } else { /* Phase 2 - Step 8: 1D no comma */
            cols = 1; /* Phase 2 - Step 8 */
        }
    }

    /* Phase 2 - Step 8: Check dtype */
    bool is_int = false; /* Phase 2 - Step 8 */
    if (strstr(header, "'<i8'") || strstr(header, "\"<i8\"") || /* Phase 2 - Step 8 */
        strstr(header, "'<i4'") || strstr(header, "\"<i4\"")) { /* Phase 2 - Step 8 */
        is_int = true; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Check Fortran order */
    bool fortran_order = false; /* Phase 2 - Step 8 */
    if (strstr(header, "'fortran_order': True") || /* Phase 2 - Step 8 */
        strstr(header, "\"fortran_order\": true")) { /* Phase 2 - Step 8 */
        fortran_order = true; /* Phase 2 - Step 8 */
    }

    free(header); /* Phase 2 - Step 8 */

    int total = rows * cols; /* Phase 2 - Step 8 */
    double *data = (double *)malloc(total * sizeof(double)); /* Phase 2 - Step 8 */

    if (is_int) { /* Phase 2 - Step 8: Read as int64 and convert */
        int64_t *idata = (int64_t *)malloc(total * sizeof(int64_t)); /* Phase 2 - Step 8 */
        {
            size_t _got = fread(idata, sizeof(int64_t), (size_t)total, fp);
            if (_got != (size_t)total) {
                fprintf(stderr, "ERROR: %s short read of int64 body (%zu/%d)\n",
                        path, _got, total);
                free(idata); free(data); fclose(fp); return NULL;
            }
        }
        for (int i = 0; i < total; i++) { /* Phase 2 - Step 8 */
            data[i] = (double)idata[i]; /* Phase 2 - Step 8 */
        }
        free(idata); /* Phase 2 - Step 8 */
    } else { /* Phase 2 - Step 8: Read as float64 directly */
        size_t _got = fread(data, sizeof(double), (size_t)total, fp); /* A7 */
        if (_got != (size_t)total) {
            fprintf(stderr, "ERROR: %s short read of float64 body (%zu/%d)\n",
                    path, _got, total);
            free(data); fclose(fp); return NULL;
        }
    }
    #undef NPY_FREAD

    /* Phase 2 - Step 8: If Fortran order, transpose to C order */
    if (fortran_order && cols > 1) { /* Phase 2 - Step 8 */
        double *transposed = (double *)malloc(total * sizeof(double)); /* Phase 2 - Step 8 */
        for (int i = 0; i < rows; i++) { /* Phase 2 - Step 8 */
            for (int j = 0; j < cols; j++) { /* Phase 2 - Step 8 */
                transposed[i * cols + j] = data[j * rows + i]; /* Phase 2 - Step 8 */
            }
        }
        free(data); /* Phase 2 - Step 8 */
        data = transposed; /* Phase 2 - Step 8 */
    }

    fclose(fp); /* Phase 2 - Step 8 */
    *out_rows = rows; /* Phase 2 - Step 8 */
    *out_cols = cols; /* Phase 2 - Step 8 */
    return data; /* Phase 2 - Step 8 */
}

/* Phase 2 - Step 8b: Read NPY as int array */
static int *read_npy_int(const char *path, int *out_n) {
    int rows, cols; /* Phase 2 - Step 8b */
    double *ddata = read_npy_f64(path, &rows, &cols); /* Phase 2 - Step 8b */
    if (!ddata) return NULL; /* Phase 2 - Step 8b */
    int total = rows * cols; /* Phase 2 - Step 8b */
    int *idata = (int *)malloc(total * sizeof(int)); /* Phase 2 - Step 8b */
    for (int i = 0; i < total; i++) { /* Phase 2 - Step 8b */
        idata[i] = (int)ddata[i]; /* Phase 2 - Step 8b */
    }
    free(ddata); /* Phase 2 - Step 8b */
    *out_n = total; /* Phase 2 - Step 8b */
    return idata; /* Phase 2 - Step 8b */
}

/* ============================================================ */
/* Phase 2 - Step 9: CSV readers                                */
/* ============================================================ */

/* Phase 2 - Step 9: Read a CSV column by name, return array */
static double *read_csv_column(const char *path, const char *col_name,
                                int *out_n) {
    FILE *fp = fopen(path, "r"); /* Phase 2 - Step 9 */
    if (!fp) { /* Phase 2 - Step 9 */
        fprintf(stderr, "ERROR: Cannot open %s\n", path); /* Phase 2 - Step 9 */
        return NULL; /* Phase 2 - Step 9 */
    }

    /* Phase 2 - Step 9: Read header line */
    char line[4096]; /* Phase 2 - Step 9 */
    if (!fgets(line, sizeof(line), fp)) { /* Phase 2 - Step 9 */
        fclose(fp); return NULL; /* Phase 2 - Step 9 */
    }

    /* Phase 2 - Step 9: Find column index */
    /* Phase 2 - Step 9: Manual CSV parse — strtok skips empty leading fields! */
    int col_idx = -1, idx = 0; /* Phase 2 - Step 9 */
    char *p_hdr = line; /* Phase 2 - Step 9 */
    while (*p_hdr && *p_hdr != '\n' && *p_hdr != '\r') { /* Phase 2 - Step 9 */
        /* Phase 2 - Step 9: Find end of current field */
        char *field_start = p_hdr; /* Phase 2 - Step 9 */
        while (*p_hdr && *p_hdr != ',' && *p_hdr != '\n' && *p_hdr != '\r') { /* Phase 2 - Step 9 */
            p_hdr++; /* Phase 2 - Step 9 */
        }
        /* Phase 2 - Step 9: Null-terminate this field temporarily */
        char saved = *p_hdr; /* Phase 2 - Step 9 */
        *p_hdr = '\0'; /* Phase 2 - Step 9 */
        /* Phase 2 - Step 9: Strip leading whitespace from field */
        while (*field_start == ' ') field_start++; /* Phase 2 - Step 9 */
        if (strcmp(field_start, col_name) == 0) { /* Phase 2 - Step 9 */
            col_idx = idx; /* Phase 2 - Step 9 */
            *p_hdr = saved; /* Phase 2 - Step 9 */
            break; /* Phase 2 - Step 9 */
        }
        *p_hdr = saved; /* Phase 2 - Step 9 */
        if (*p_hdr == ',') p_hdr++; /* Phase 2 - Step 9: skip comma */
        idx++; /* Phase 2 - Step 9 */
    }

    if (col_idx < 0) { /* Phase 2 - Step 9 */
        fprintf(stderr, "ERROR: Column '%s' not found in %s\n", col_name, path); /* Phase 2 - Step 9 */
        fclose(fp); return NULL; /* Phase 2 - Step 9 */
    }

    /* Phase 2 - Step 9: Count rows */
    int capacity = 1024; /* Phase 2 - Step 9 */
    double *data = (double *)malloc(capacity * sizeof(double)); /* Phase 2 - Step 9 */
    int n = 0; /* Phase 2 - Step 9 */

    while (fgets(line, sizeof(line), fp)) { /* Phase 2 - Step 9 */
        if (line[0] == '\n' || line[0] == '\r') continue; /* Phase 2 - Step 9 */
        /* Phase 2 - Step 9: Walk to correct column */
        char *p = line; /* Phase 2 - Step 9 */
        for (int i = 0; i < col_idx; i++) { /* Phase 2 - Step 9 */
            p = strchr(p, ','); /* Phase 2 - Step 9 */
            if (!p) break; /* Phase 2 - Step 9 */
            p++; /* Phase 2 - Step 9 */
        }
        if (!p) continue; /* Phase 2 - Step 9 */

        if (n >= capacity) { /* Phase 2 - Step 9 */
            capacity *= 2; /* Phase 2 - Step 9 */
            data = (double *)realloc(data, capacity * sizeof(double)); /* Phase 2 - Step 9 */
        }
        data[n++] = atof(p); /* Phase 2 - Step 9 */
    }

    fclose(fp); /* Phase 2 - Step 9 */
    *out_n = n; /* Phase 2 - Step 9 */
    return data; /* Phase 2 - Step 9 */
}

/* Phase 2 - Step 9b: Read CSV column as int */
static int *read_csv_column_int(const char *path, const char *col_name,
                                 int *out_n) {
    int n; /* Phase 2 - Step 9b */
    double *ddata = read_csv_column(path, col_name, &n); /* Phase 2 - Step 9b */
    if (!ddata) return NULL; /* Phase 2 - Step 9b */
    int *idata = (int *)malloc(n * sizeof(int)); /* Phase 2 - Step 9b */
    for (int i = 0; i < n; i++) { /* Phase 2 - Step 9b */
        idata[i] = (int)ddata[i]; /* Phase 2 - Step 9b */
    }
    free(ddata); /* Phase 2 - Step 9b */
    *out_n = n; /* Phase 2 - Step 9b */
    return idata; /* Phase 2 - Step 9b */
}

/* A2-06 V4 section 4: bind the original configuration label to every loaded
 * level without retaining the label itself in the physics object.  The fixed
 * digest table is diagnostic-only and is consumed by the A_ul crosswalk. */
static char (*read_csv_column_sha256(const char *path, const char *col_name,
                                     int *out_n))[65]
{
    FILE *fp = fopen(path, "r");
    char line[4096];
    int col_idx = -1, idx = 0, n = 0, cap = 1024;
    char (*hashes)[65] = NULL;
    if (!fp || !fgets(line, sizeof(line), fp)) {
        if (fp) fclose(fp);
        return NULL;
    }
    for (char *p = line; *p && *p != '\n' && *p != '\r'; idx++) {
        char *start = p;
        while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
        char saved = *p;
        *p = '\0';
        while (*start == ' ') start++;
        if (strcmp(start, col_name) == 0) col_idx = idx;
        *p = saved;
        if (col_idx >= 0 || *p != ',') break;
        p++;
    }
    if (col_idx < 0) { fclose(fp); return NULL; }
    hashes = (char (*)[65])malloc((size_t)cap * sizeof(*hashes));
    if (!hashes) { fclose(fp); return NULL; }
    while (fgets(line, sizeof(line), fp)) {
        char *p = line, *end;
        char decoded[4096];
        size_t decoded_n = 0;
        if (line[0] == '\n' || line[0] == '\r') continue;
        for (int c = 0; c <= col_idx; c++) {
            int quoted = (*p == '"');
            if (quoted) p++;
            decoded_n = 0;
            while (*p) {
                if (quoted && *p == '"') {
                    if (p[1] == '"') {
                        if (decoded_n + 1 >= sizeof(decoded)) break;
                        decoded[decoded_n++] = '"';
                        p += 2;
                        continue;
                    }
                    p++;
                    while (*p == ' ' || *p == '\t') p++;
                    break;
                }
                if (!quoted && (*p == ',' || *p == '\n' || *p == '\r')) break;
                if (decoded_n + 1 >= sizeof(decoded)) break;
                decoded[decoded_n++] = *p++;
            }
            if (decoded_n + 1 >= sizeof(decoded)) {
                free(hashes); fclose(fp); return NULL;
            }
            decoded[decoded_n] = '\0';
            if (c == col_idx) break;
            while (*p && *p != ',') p++;
            if (*p != ',') { p = NULL; break; }
            p++;
        }
        if (!p) { free(hashes); fclose(fp); return NULL; }
        p = decoded;
        end = decoded + decoded_n;
        while (p < end && (*p == ' ' || *p == '\t')) p++;
        while (end > p && (end[-1] == ' ' || end[-1] == '\t')) end--;
        if (end == p) { free(hashes); fclose(fp); return NULL; }
        if (n == cap) {
            cap *= 2;
            void *grown = realloc(hashes, (size_t)cap * sizeof(*hashes));
            if (!grown) { free(hashes); fclose(fp); return NULL; }
            hashes = (char (*)[65])grown;
        }
        KShapeSHA256 sha;
        unsigned char digest[32];
        kshape_sha256_init(&sha);
        kshape_sha256_update(&sha, p, (size_t)(end - p));
        kshape_sha256_final(&sha, digest);
        for (int b = 0; b < 32; b++)
            snprintf(hashes[n] + 2*b, 3, "%02x", digest[b]);
        hashes[n][64] = '\0';
        n++;
    }
    fclose(fp);
    *out_n = n;
    return hashes;
}

/* CONFIG-PREC: config.json owns the deck declaration; plasma_state.csv is a
 * consistency witness, never an override.  Runtime overrides have to be
 * explicit and are logged below.  Keep the tolerances here (rather than in a
 * deck) so a stale deck cannot relax its own integrity check. */
#define CONFIG_PREC_T_DECL_ABS_TOL_K 5.0
#define CONFIG_PREC_T_PROFILE_ABS_TOL_K 0.01
#define CONFIG_PREC_T_REL_TOL 1.0e-9

static int config_prec_parse_switch(const char *name, int *enabled) {
    const char *value = getenv(name);
    if (!value) {
        *enabled = 0;
        return 0;
    }
    if (strcmp(value, "0") == 0) {
        *enabled = 0;
        return 0;
    }
    if (strcmp(value, "1") == 0) {
        *enabled = 1;
        return 0;
    }
    fprintf(stderr,
            "[CONFIG-PREC][FATAL] %s='%s' is invalid; expected exactly 0 or 1\n",
            name, value);
    return -1;
}

static int config_prec_parse_positive_double(const char *name,
                                             const char *value,
                                             double *parsed) {
    char *end = NULL;
    errno = 0;
    double result = strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' ||
        !isfinite(result) || result <= 0.0) {
        fprintf(stderr,
                "[CONFIG-PREC][FATAL] %s='%s' is not a finite positive number\n",
                name, value);
        return -1;
    }
    *parsed = result;
    return 0;
}

typedef struct {
    int rows;
    int invalid_rows;
    double color_first;
    double color_min;
    double color_max;
} ConfigPrecWitness;

/* Read the retired scalar columns only as a deck-integrity witness.  Values
 * are reduced row-by-row and are never published into PlasmaState, retained,
 * or used to seed radiation/material state. */
static int config_prec_read_witness(const char *ref_dir,
                                    ConfigPrecWitness *witness) {
    char path[512];
    char line[4096];
    int w_col = -1, trad_col = -1;
    snprintf(path, sizeof(path), "%s/plasma_state.csv", ref_dir);
    FILE *fp = fopen(path, "r");
    if (!fp || !fgets(line, sizeof(line), fp)) {
        if (fp) fclose(fp);
        fprintf(stderr,
                "[CONFIG-PREC][FATAL] plasma_state.csv witness unavailable in %s\n",
                ref_dir);
        return -1;
    }

    int column = 0;
    for (char *field = line; field && *field; column++) {
        char *end = strpbrk(field, ",\r\n");
        char saved = end ? *end : '\0';
        if (end) *end = '\0';
        while (*field == ' ' || *field == '\t') field++;
        if (strcmp(field, "W") == 0) w_col = column;
        if (strcmp(field, "T_rad") == 0) trad_col = column;
        if (!end || saved != ',') break;
        *end = saved;
        field = end + 1;
    }
    if (w_col < 0 || trad_col < 0) {
        fclose(fp);
        fprintf(stderr,
                "[CONFIG-PREC][FATAL] plasma_state.csv witness columns unavailable in %s\n",
                ref_dir);
        return -1;
    }

    memset(witness, 0, sizeof(*witness));
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        double row_w = NAN, row_trad = NAN;
        int row_column = 0;
        for (char *field = line; field && *field; row_column++) {
            char *end = strpbrk(field, ",\r\n");
            char saved = end ? *end : '\0';
            if (end) *end = '\0';
            if (row_column == w_col || row_column == trad_col) {
                char *number_end = NULL;
                errno = 0;
                double value = strtod(field, &number_end);
                while (number_end && (*number_end == ' ' || *number_end == '\t'))
                    number_end++;
                if (errno == ERANGE || number_end == field ||
                    (number_end && *number_end != '\0'))
                    value = NAN;
                if (row_column == w_col) row_w = value;
                if (row_column == trad_col) row_trad = value;
            }
            if (!end || saved != ',') break;
            *end = saved;
            field = end + 1;
        }

        witness->rows++;
        if (!isfinite(row_w) || row_w <= 0.0 || row_w > 1.0 ||
            !isfinite(row_trad) || row_trad <= 0.0) {
            witness->invalid_rows++;
            continue;
        }
        double color = row_trad / pow(row_w, 0.25);
        if (!isfinite(color) || color <= 0.0) {
            witness->invalid_rows++;
            continue;
        }
        if (witness->rows - witness->invalid_rows == 1) {
            witness->color_first = witness->color_min = witness->color_max = color;
        } else {
            if (color < witness->color_min) witness->color_min = color;
            if (color > witness->color_max) witness->color_max = color;
        }
    }
    if (ferror(fp) || fclose(fp) != 0) {
        fprintf(stderr,
                "[CONFIG-PREC][FATAL] plasma_state.csv witness read failed in %s\n",
                ref_dir);
        return -1;
    }
    return 0;
}

static int config_prec_resolve_boundary_temperature(
        const char *ref_dir, double deck_T_inner,
        int expected_shells, double *effective_T_inner) {
    int strict = 0;
    if (config_prec_parse_switch("LUMINA_CONFIG_PREC", &strict) != 0)
        return -1;

    printf("  [CONFIG-PREC] priority=argv>env>config.json>compiled-default; "
           "plasma_state.csv=integrity-witness-only; native J_nu seed owns "
           "radiation initialization; gate=%s\n",
           strict ? "ON" : "OFF");
    if (!isfinite(deck_T_inner) || deck_T_inner <= 0.0) {
        fprintf(stderr,
                "[CONFIG-PREC][FATAL] config.json:T_inner_K=%.17g is not "
                "finite and positive\n", deck_T_inner);
        return -1;
    }

    ConfigPrecWitness witness;
    if (config_prec_read_witness(ref_dir, &witness) != 0) return -1;
    int valid_rows = witness.rows - witness.invalid_rows;
    double spread = valid_rows > 0
        ? witness.color_max - witness.color_min : INFINITY;
    double profile_tol = valid_rows > 0
        ? CONFIG_PREC_T_PROFILE_ABS_TOL_K +
          CONFIG_PREC_T_REL_TOL *
              fmax(fabs(witness.color_min), fabs(witness.color_max))
        : 0.0;
    double delta = valid_rows > 0
        ? fabs(deck_T_inner - witness.color_first) : INFINITY;
    double decl_tol = CONFIG_PREC_T_DECL_ABS_TOL_K +
        CONFIG_PREC_T_REL_TOL *
            fmax(fabs(deck_T_inner), fabs(witness.color_first));
    double rel = valid_rows > 0 ? delta / fabs(deck_T_inner) : INFINITY;
    int violation = witness.invalid_rows != 0 ||
                    witness.rows != expected_shells ||
                    spread > profile_tol || delta > decl_tol;

    if (valid_rows > 0) {
        printf("  [CONFIG-PREC] deck=%s config.json:T_inner_K=%.9f K; "
               "plasma inferred-color=%.9f K; spread=%.9g K; "
               "delta=%.9f K (%.6f%%); limits(decl/profile)=%.6g/%.6g K\n",
               ref_dir, deck_T_inner, witness.color_first, spread, delta,
               100.0 * rel, decl_tol, profile_tol);
    } else {
        printf("  [CONFIG-PREC] deck=%s plasma inferred-color=unavailable\n",
               ref_dir);
    }
    if (violation) {
        FILE *stream = strict ? stderr : stdout;
        fprintf(stream,
                "[CONFIG-PREC][%s] boundary-temperature declarations disagree "
                "or are not certifiable: invalid_rows=%d color_rows=%d/%d\n",
                strict ? "FATAL" : "WARN", witness.invalid_rows, valid_rows,
                witness.rows);
        if (strict) return -1;
    } else {
        printf("  [CONFIG-PREC][PASS] boundary-temperature declarations agree\n");
    }

    *effective_T_inner = deck_T_inner;
    const char *override = getenv("LUMINA_T_INNER_FIX");
    if (override) {
        if (config_prec_parse_positive_double("LUMINA_T_INNER_FIX", override,
                                              effective_T_inner) != 0)
            return -1;
        printf("  [CONFIG-PREC] effective T_inner=%.9f K "
               "source=env:LUMINA_T_INNER_FIX "
               "(does not waive deck-integrity result)\n",
               *effective_T_inner);
    } else {
        printf("  [CONFIG-PREC] effective T_inner=%.9f K "
               "source=config.json:T_inner_K (argv channel not defined)\n",
               *effective_T_inner);
    }
    return 0;
}

/* ============================================================ */
/* Phase 2 - Step 10: Main data loader                          */
/* ============================================================ */

int load_tardis_reference_data(const char *ref_dir, Geometry *geo,
                                OpacityState *opacity, PlasmaState *plasma,
                                MCConfig *config) {
    char path[512]; /* Phase 2 - Step 10 */
    int n; /* Phase 2 - Step 10 */
    double config_prec_deck_T_inner = 0.0;

    if (seed_capability_reject_obsolete_options() != SEED_OK)
        return -1;
    {
        static const char *const retired_scalar_options[] = {
            "LUMINA_FIXED_TRAD_PROFILE",
            "LUMINA_W_CAP",
            "LUMINA_VALIDATE_PLASMA",
            "LUMINA_OUTER_TE_DAMP_FACTOR",
            "LUMINA_OUTER_TE_DAMP_SMIN",
            "LUMINA_F_COLL_BOOST",
            "LUMINA_KPEMISS_BSRC_TAU",
            "LUMINA_BSRC_WFLOOR",
            "LUMINA_CMF_EPAY_HOTF"
        };
        for (size_t i = 0;
             i < sizeof(retired_scalar_options) / sizeof(retired_scalar_options[0]);
             ++i) {
            if (getenv(retired_scalar_options[i]) != NULL) {
                fprintf(stderr,
                        "BLOCKED_OBSOLETE_SCALAR_OPTION: %s was removed by A2-17\n",
                        retired_scalar_options[i]);
                return -1;
            }
        }
        const char *epay = getenv("LUMINA_CMF_EPAY");
        if (epay != NULL && atoi(epay) >= 2) {
            fprintf(stderr,
                    "BLOCKED_OBSOLETE_SCALAR_OPTION: LUMINA_CMF_EPAY>=2 "
                    "requires the retired scalar hot/cold classifier\n");
            return -1;
        }
    }

    /* Optional Wave-1 field is NULL unless its gate explicitly requests it.
     * The legacy loaders use a stack PlasmaState, so initialize the new member
     * here rather than relying on caller zeroing. */
    plasma->clump_factor = NULL;

    printf("Loading TARDIS reference data from %s...\n", ref_dir); /* Phase 2 - Step 10 */

    /* Phase 2 - Step 10a: Load geometry */
    snprintf(path, sizeof(path), "%s/geometry.csv", ref_dir); /* Phase 2 - Step 10a */
    geo->r_inner = read_csv_column(path, "r_inner", &n); /* Phase 2 - Step 10a */
    geo->n_shells = n; /* Phase 2 - Step 10a */
    geo->r_outer = read_csv_column(path, "r_outer", &n); /* Phase 2 - Step 10a */
    geo->v_inner = read_csv_column(path, "v_inner", &n); /* Phase 2 - Step 10a */
    geo->v_outer = read_csv_column(path, "v_outer", &n); /* Phase 2 - Step 10a */
    printf("  Geometry: %d shells\n", geo->n_shells); /* Phase 2 - Step 10a */
    printf("    r_inner[0] = %.6e cm, r_outer[%d] = %.6e cm\n", /* Phase 2 - Step 10a */
           geo->r_inner[0], n - 1, geo->r_outer[n - 1]); /* Phase 2 - Step 10a */
    printf("    v_inner[0] = %.6e cm/s, v_outer[%d] = %.6e cm/s\n", /* Phase 2 - Step 10a */
           geo->v_inner[0], n - 1, geo->v_outer[n - 1]); /* Phase 2 - Step 10a */

    /* Phase 2 - Step 10b: Load config */
    snprintf(path, sizeof(path), "%s/config.json", ref_dir); /* Phase 2 - Step 10b */
    FILE *fp = fopen(path, "r"); /* Phase 2 - Step 10b */
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return -1;
    }
    {
        char buf[4096]; /* Phase 2 - Step 10b */
        size_t nr = fread(buf, 1, sizeof(buf) - 1, fp); /* Phase 2 - Step 10b */
        buf[nr] = '\0'; /* Phase 2 - Step 10b */
        fclose(fp); /* Phase 2 - Step 10b */

        /* A5: track which required keys are present.  Previously missing keys
         * silently left the struct zeroed (T_inner=0, n_packets=0 etc.) and the
         * run proceeded with nonsense — would show up as "T_inner: 0.00 K" in
         * the banner.  Now: require all 6, abort on first missing. */
        char *p; /* Phase 2 - Step 10b */
        int missing = 0;
        #define PARSE_REQ(key, sink, conv)                                  \
            do {                                                             \
                p = strstr(buf, "\"" key "\"");                              \
                if (!p || !(p = strchr(p, ':'))) {                           \
                    fprintf(stderr,                                          \
                        "ERROR: %s missing required key \"%s\"\n",           \
                        path, key);                                          \
                    missing = 1;                                             \
                } else {                                                     \
                    sink = conv(p + 1);                                      \
                }                                                            \
            } while (0)
        PARSE_REQ("time_explosion_s",       geo->time_explosion,         atof);
        PARSE_REQ("T_inner_K",              config->T_inner,             atof);
        PARSE_REQ("luminosity_inner_erg_s", config->luminosity_requested,atof);
        PARSE_REQ("n_packets",              config->n_packets,           atoi);
        PARSE_REQ("n_iterations",           config->n_iterations,        atoi);
        PARSE_REQ("seed",                   config->seed,                (uint64_t)atol);
        #undef PARSE_REQ
        if (missing) return -1;
        config_prec_deck_T_inner = config->T_inner;

        printf("  Config: t_exp=%.6e s, T_inner=%.2f K, L=%.3e erg/s\n", /* Phase 2 - Step 10b */
               geo->time_explosion, config->T_inner, config->luminosity_requested); /* Phase 2 - Step 10b */
        printf("    n_packets=%d, n_iter=%d (config-file defaults, pre-override;"
               " effective values in 'Simulation parameters' below), seed=%lu\n",
               config->n_packets, config->n_iterations, config->seed);
    }

    /* Phase 2 - Step 10c: Load electron densities */
    snprintf(path, sizeof(path), "%s/electron_densities.csv", ref_dir); /* Phase 2 - Step 10c */
    opacity->electron_density = read_csv_column(path, "n_e", &n); /* Phase 2 - Step 10c */
    printf("  Electron densities: n_e[0]=%.6e, n_e[%d]=%.6e cm^-3\n", /* Phase 2 - Step 10c */
           opacity->electron_density[0], n - 1, opacity->electron_density[n - 1]); /* Phase 2 - Step 10c */

    /* A2-17: runtime never opens legacy scalar columns. Native J_nu loading is
     * owned by jnu_seed.c and the generation-zero seed capability. */
    plasma->n_shells = geo->n_shells;
    if (config_prec_resolve_boundary_temperature(ref_dir,
            config_prec_deck_T_inner, geo->n_shells, &config->T_inner) != 0)
        return -1;

    /* Phase 2 - Step 10d2: Load density */
    snprintf(path, sizeof(path), "%s/density.csv", ref_dir); /* Phase 2 - Step 10d2 */
    plasma->rho = read_csv_column(path, "rho", &n); /* Phase 2 - Step 10d2 */

    /* ARTIS calculate_chi_bf_gammacontr uses clumpednne = nne*clumpfactor.
     * Keep the field absent on the gate-OFF path. On the repair path accept a
     * per-shell `clumping_factors.csv:clump_factor`, with an optional scalar
     * LUMINA_BF_CLUMP_FACTOR override. Smooth models default exactly to 1. */
    {
        const char *stim = getenv("LUMINA_FIX_BF_STIM_RECOMB");
        if (stim && atoi(stim) != 0) {
            int ns = geo->n_shells;
            plasma->clump_factor = (double *)malloc((size_t)ns * sizeof(double));
            for (int s = 0; s < ns; s++) plasma->clump_factor[s] = 1.0;

            snprintf(path, sizeof(path), "%s/clumping_factors.csv", ref_dir);
            FILE *cf = fopen(path, "r");
            if (cf) {
                fclose(cf);
                int nc = 0;
                double *loaded = read_csv_column(path, "clump_factor", &nc);
                if (loaded && nc == ns) {
                    for (int s = 0; s < ns; s++)
                        if (loaded[s] > 0.0 && isfinite(loaded[s]))
                            plasma->clump_factor[s] = loaded[s];
                }
                free(loaded);
            }
            const char *scalar = getenv("LUMINA_BF_CLUMP_FACTOR");
            if (scalar && atof(scalar) > 0.0 && isfinite(atof(scalar))) {
                double f = atof(scalar);
                for (int s = 0; s < ns; s++) plasma->clump_factor[s] = f;
            }
        }
    }

    /* Generation-zero material-temperature seed; radiation is not consulted. */
    opacity->t_electrons = (double *)malloc(geo->n_shells * sizeof(double)); /* Phase 2 - Step 10d3 */
    for (int i = 0; i < geo->n_shells; i++) { /* Phase 2 - Step 10d3 */
        opacity->t_electrons[i] = config->T_inner;
    }

    /* Phase 2 - Step 10e: Load line list (nu, sorted descending) */
    snprintf(path, sizeof(path), "%s/line_list.csv", ref_dir); /* Phase 2 - Step 10e */
    opacity->line_list_nu = read_csv_column(path, "nu", &n); /* Phase 2 - Step 10e */
    opacity->n_lines = n; /* Phase 2 - Step 10e */
    opacity->n_shells = geo->n_shells; /* Phase 2 - Step 10e */
    printf("  Lines: %d total, nu[0]=%.6e Hz (%.1f A), nu[%d]=%.6e Hz (%.1f A)\n", /* Phase 2 - Step 10e */
           n, opacity->line_list_nu[0], /* Phase 2 - Step 10e */
           C_SPEED_OF_LIGHT / opacity->line_list_nu[0] * 1e8, /* Phase 2 - Step 10e */
           n - 1, opacity->line_list_nu[n - 1], /* Phase 2 - Step 10e */
           C_SPEED_OF_LIGHT / opacity->line_list_nu[n - 1] * 1e8); /* Phase 2 - Step 10e */

    /* Task #072: Store line_list.csv path for later atomic data loading */
    /* (line_atomic_number etc. loaded in load_atomic_data) */

    /* A9: Verify strictly non-ascending order. Sobolev binary search assumes
     * descending nu; a single out-of-order pair silently mis-routes packets
     * onto the wrong line and warps every per-band ratio. Abort hard. */
    int desc_violations = 0;
    int first_bad_i = -1;
    for (int i = 1; i < n; i++) {
        if (opacity->line_list_nu[i] > opacity->line_list_nu[i - 1]) {
            if (first_bad_i < 0) first_bad_i = i;
            desc_violations++;
        }
    }
    if (desc_violations > 0) {
        fprintf(stderr,
                "FATAL: line_list_nu not descending — %d violation(s), first at i=%d "
                "(nu[%d]=%.6e > nu[%d]=%.6e). Sobolev binary search REQUIRES descending.\n"
                "  Regenerate reference with sort_values('nu', ascending=False, kind='stable').\n",
                desc_violations, first_bad_i,
                first_bad_i, opacity->line_list_nu[first_bad_i],
                first_bad_i - 1, opacity->line_list_nu[first_bad_i - 1]);
        return -1;
    }
    printf("  Line order: DESCENDING (correct, %d pairs)\n", n - 1);

    /* K-SHAPE: establish the transition row authority from macro_atom_data.csv
     * before either NPY can influence runtime dimensions. */
    snprintf(path, sizeof(path), "%s/macro_atom_data.csv", ref_dir);
    int nt=0, nd=0, nl=0;
    opacity->transition_type = read_csv_column_int(path, "transition_type", &nt);
    opacity->destination_level_id = read_csv_column_int(path, "destination_level_idx", &nd);
    opacity->transition_line_id = read_csv_column_int(path, "lines_idx", &nl);
    if (!opacity->transition_type || !opacity->destination_level_id ||
        !opacity->transition_line_id || nt <= 0 || nt != nd || nt != nl) {
        fprintf(stderr,
                "[K-SHAPE][FATAL] macro_atom_data row contract failed "
                "(transition_type=%d destination=%d lines_idx=%d)\n", nt, nd, nl);
        return -1;
    }
    opacity->n_macro_transitions = nt;

    if (validate_kshape_contract(ref_dir, opacity->n_lines,
                                 opacity->n_macro_transitions,
                                 opacity->n_shells) != 0)
        return -1;

    /* Phase 2 - Step 10f: Load tau_sobolev [n_lines, n_shells].  This disk
     * value is a shape/epoch-checked seed only; K-FRESH marks it stale below. */
    snprintf(path, sizeof(path), "%s/tau_sobolev.npy", ref_dir); /* Phase 2 - Step 10f */
    int tr=0, tc=0; /* Phase 2 - Step 10f */
    opacity->tau_sobolev = read_npy_f64_strict_2d(path, &tr, &tc);
    printf("  tau_sobolev: [%d x %d] (expect [%d x %d])\n", /* Phase 2 - Step 10f */
           tr, tc, opacity->n_lines, opacity->n_shells); /* Phase 2 - Step 10f */
    if (!opacity->tau_sobolev || tr != opacity->n_lines || tc != opacity->n_shells) {
        fprintf(stderr, "[K-SHAPE][FATAL] tau_sobolev [%d x %d] != expected [%d x %d]\n",
                tr, tc, opacity->n_lines, opacity->n_shells);
        free(opacity->tau_sobolev);
        opacity->tau_sobolev = NULL;
        return -1;
    }

    /* The solver, not the deck, owns physical tau.  Generation zero is the
     * validated-but-stale disk seed and must never reach a physics consumer. */
    opacity->tau_required_generation = 1;
    opacity->tau_computed_generation = 0;
    opacity->tau_first_consumer_generation = 0;

    /* CMF: per-line NLTE source function, populated during plasma/NLTE update.
     * 0 (calloc default) signals "use fallback" in the CMF solver. */
    opacity->line_source_S = (double *)calloc((size_t)opacity->n_lines * opacity->n_shells, sizeof(double));
    opacity->tau_validity = (A208Validity *)calloc(
        (size_t)opacity->n_lines * opacity->n_shells, sizeof(A208Validity));
    opacity->line_source_validity = (A208Validity *)calloc(
        (size_t)opacity->n_lines * opacity->n_shells, sizeof(A208Validity));

    /* Phase 2 - Step 10g: Load transition probabilities [n_trans, n_shells] */
    snprintf(path, sizeof(path), "%s/transition_probabilities.npy", ref_dir); /* Phase 2 - Step 10g */
    opacity->transition_probabilities = read_npy_f64_strict_2d(path, &tr, &tc);
    printf("  transition_probabilities: [%d x %d]\n", tr, tc); /* Phase 2 - Step 10g */
    if (!opacity->transition_probabilities || tr != opacity->n_macro_transitions ||
        tc != opacity->n_shells) {
        fprintf(stderr,
                "[K-SHAPE][FATAL] transition_probabilities [%d x %d] != expected [%d x %d]\n",
                tr, tc, opacity->n_macro_transitions, opacity->n_shells);
        free(opacity->transition_probabilities);
        opacity->transition_probabilities = NULL;
        return -1;
    }

    /* Phase 2 - Step 10h: Load macro-atom references */
    snprintf(path, sizeof(path), "%s/macro_atom_references.csv", ref_dir); /* Phase 2 - Step 10h */
    int *block_refs = read_csv_column_int(path, "block_references", &n); /* Phase 2 - Step 10h */
    opacity->n_macro_levels = n; /* Phase 2 - Step 10h */
    /* Phase 2 - Step 10h: Build block_references array [n_levels + 1] */
    opacity->macro_block_references = (int *)malloc((n + 1) * sizeof(int)); /* Phase 2 - Step 10h */
    for (int i = 0; i < n; i++) { /* Phase 2 - Step 10h */
        opacity->macro_block_references[i] = block_refs[i]; /* Phase 2 - Step 10h */
    }
    opacity->macro_block_references[n] = opacity->n_macro_transitions; /* Phase 2 - Step 10h */
    free(block_refs); /* Phase 2 - Step 10h */
    printf("  Macro-atom: %d levels, %d transitions\n", /* Phase 2 - Step 10h */
           opacity->n_macro_levels, opacity->n_macro_transitions); /* Phase 2 - Step 10h */

    /* Phase 2 - Step 10i: transition columns were loaded before the NPYs so
     * their CSV row count, not an untrusted array header, is authoritative. */
    printf("  Macro transitions loaded: %d entries\n", opacity->n_macro_transitions);

    /* Phase 2 - Step 10j: Load line2macro_level_upper */
    snprintf(path, sizeof(path), "%s/line2macro_level_upper.npy", ref_dir); /* Phase 2 - Step 10j */
    opacity->line2macro_level_upper = read_npy_int(path, &n); /* Phase 2 - Step 10j */
    printf("  line2macro_level_upper: %d entries\n", n); /* Phase 2 - Step 10j */

    /* k-packet tables: lazily built by compute_transition_probabilities when
     * LUMINA_KPACKET is enabled; NULL until then. */
    opacity->p_kpacket = NULL;
    opacity->kpacket_cdf = NULL;

    /* bf recomb-cascade topology: lazily built by compute_transition_probabilities
     * when LUMINA_MACROATOM_BF is enabled; NULL/0 until then. */
    opacity->recomb_block_refs = NULL;
    opacity->recomb_dest_level = NULL;
    opacity->recomb_nu_edge    = NULL;
    opacity->recomb_is_emit    = NULL;
    opacity->recomb_prob       = NULL;
    opacity->n_recomb          = 0;
    opacity->recomb_sigma_edge = NULL;   /* [MA-RADRECOMB tau-gate] built under the rr gate */
    opacity->recomb_emit_shell = NULL;   /* [MA-RADRECOMB tau-gate] per-shell emit decision */
    opacity->iup_dest_level    = NULL;   /* [MA-RADRECOMB iup] lazily built under the gate */
    opacity->iup_prob          = NULL;
    opacity->chi_ff_nnionpart  = NULL;   /* [ARTIS-PARITY D5] lazily built under parity */

    printf("Data loading complete.\n"); /* Phase 2 - Step 10 */
    return 0; /* Phase 2 - Step 10 */
}

/* ============================================================ */
/* Phase 2 - Step 11: Memory management                         */
/* ============================================================ */

void free_geometry(Geometry *geo) { /* Phase 2 - Step 11 */
    free(geo->r_inner); /* Phase 2 - Step 11 */
    free(geo->r_outer); /* Phase 2 - Step 11 */
    free(geo->v_inner); /* Phase 2 - Step 11 */
    free(geo->v_outer); /* Phase 2 - Step 11 */
}

void free_opacity_state(OpacityState *op) { /* Phase 2 - Step 11 */
    free(op->line_list_nu); /* Phase 2 - Step 11 */
    free(op->tau_sobolev); /* Phase 2 - Step 11 */
    free(op->line_source_S);
    free(op->tau_validity);
    free(op->line_source_validity);
    a208_publication_free(&op->cpu_opacity);
    a209_publication_free(&op->cpu_emissivity);
    free(op->electron_density); /* Phase 2 - Step 11 */
    free(op->t_electrons); /* Phase 2 - Step 11 */
    free(op->macro_block_references); /* Phase 2 - Step 11 */
    free(op->transition_type); /* Phase 2 - Step 11 */
    free(op->destination_level_id); /* Phase 2 - Step 11 */
    free(op->transition_line_id); /* Phase 2 - Step 11 */
    free(op->transition_probabilities); /* Phase 2 - Step 11 */
    free(op->line2macro_level_upper); /* Phase 2 - Step 11 */
    free(op->recomb_block_refs);      /* bf recomb cascade (NULL-safe) */
    free(op->recomb_dest_level);
    free(op->recomb_nu_edge);
    free(op->recomb_is_emit);
    free(op->recomb_prob);
    free(op->recomb_sigma_edge);      /* [MA-RADRECOMB tau-gate] (NULL-safe) */
    free(op->recomb_emit_shell);      /* [MA-RADRECOMB tau-gate] (NULL-safe) */
    free(op->iup_dest_level);         /* [MA-RADRECOMB iup] (NULL-safe) */
    free(op->iup_prob);
    free(op->chi_ff_nnionpart);       /* [ARTIS-PARITY D5] (NULL-safe) */
}

void free_plasma_state(PlasmaState *ps) { /* Phase 2 - Step 11 */
    free(ps->rho); /* Phase 2 - Step 11 */
    free(ps->n_electron); /* Task #072 */
    free(ps->clump_factor); /* Wave-1 BF; NULL-safe and absent on gate OFF */
    free(ps->T_e); /* P6: per-shell electron temperature */
    a210_publication_free(&ps->te_publication);
}

Estimators *create_estimators(int n_shells, int n_lines) { /* Phase 2 - Step 11 */
    Estimators *est = (Estimators *)calloc(1, sizeof(Estimators)); /* Phase 2 - Step 11 */
    est->n_shells = n_shells; /* Phase 2 - Step 11 */
    est->n_lines = n_lines; /* Phase 2 - Step 11 */
    est->j_estimator = (double *)calloc(n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    est->nu_bar_estimator = (double *)calloc(n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    est->j_blue_estimator = (double *)calloc((size_t)n_lines * n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    est->Edotlu_estimator = (double *)calloc((size_t)n_lines * n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    return est; /* Phase 2 - Step 11 */
}

void reset_estimators(Estimators *est) { /* Phase 2 - Step 11 */
    memset(est->j_estimator, 0, est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
    memset(est->nu_bar_estimator, 0, est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
    memset(est->j_blue_estimator, 0, (size_t)est->n_lines * est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
    memset(est->Edotlu_estimator, 0, (size_t)est->n_lines * est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
}

void free_estimators(Estimators *est) { /* Phase 2 - Step 11 */
    free(est->j_estimator); /* Phase 2 - Step 11 */
    free(est->nu_bar_estimator); /* Phase 2 - Step 11 */
    free(est->j_blue_estimator); /* Phase 2 - Step 11 */
    free(est->Edotlu_estimator); /* Phase 2 - Step 11 */
    free(est); /* Phase 2 - Step 11 */
}

Spectrum *create_spectrum(double lambda_min, double lambda_max, int n_bins) { /* Phase 2 - Step 11 */
    Spectrum *spec = (Spectrum *)calloc(1, sizeof(Spectrum)); /* Phase 2 - Step 11 */
    spec->n_bins = n_bins; /* Phase 2 - Step 11 */
    spec->lambda_min = lambda_min; /* Phase 2 - Step 11 */
    spec->lambda_max = lambda_max; /* Phase 2 - Step 11 */
    spec->flux = (double *)calloc(n_bins, sizeof(double)); /* Phase 2 - Step 11 */
    spec->wavelength = (double *)malloc(n_bins * sizeof(double)); /* Phase 2 - Step 11 */
    double dlambda = (lambda_max - lambda_min) / n_bins; /* Phase 2 - Step 11 */
    for (int i = 0; i < n_bins; i++) { /* Phase 2 - Step 11 */
        spec->wavelength[i] = lambda_min + (i + 0.5) * dlambda; /* Phase 2 - Step 11 */
    }
    return spec; /* Phase 2 - Step 11 */
}

void reset_spectrum(Spectrum *spec) { /* Phase 2 - Step 11 */
    memset(spec->flux, 0, spec->n_bins * sizeof(double)); /* Phase 2 - Step 11 */
}

void free_spectrum(Spectrum *spec) { /* Phase 2 - Step 11 */
    free(spec->flux); /* Phase 2 - Step 11 */
    free(spec->wavelength); /* Phase 2 - Step 11 */
    free(spec); /* Phase 2 - Step 11 */
}

/* ============================================================ */
/* Task #072: Load atomic data for plasma solver                 */
/* ============================================================ */

int load_atomic_data(AtomicData *atom, const char *ref_dir, int n_shells) {
    char path[512];
    int n;

    memset(atom, 0, sizeof(AtomicData));
    printf("\nLoading atomic data for plasma solver...\n");

    /* --- Line columns from line_list.csv --- */
    snprintf(path, sizeof(path), "%s/line_list.csv", ref_dir);
    atom->line_atomic_number = read_csv_column_int(path, "atomic_number", &n);
    atom->line_ion_number    = read_csv_column_int(path, "ion_number", &n);
    atom->line_level_lower   = read_csv_column_int(path, "level_number_lower", &n);
    atom->line_level_upper   = read_csv_column_int(path, "level_number_upper", &n);
    atom->line_f_lu          = read_csv_column(path, "f_lu", &n);
    atom->line_wavelength_cm = read_csv_column(path, "wavelength_cm", &n);

    /* NLTE: Einstein coefficients and line frequencies */
    atom->line_A_ul = read_csv_column(path, "A_ul", &n);
    atom->line_B_lu = read_csv_column(path, "B_lu", &n);
    atom->line_B_ul = read_csv_column(path, "B_ul", &n);
    atom->line_nu   = read_csv_column(path, "nu", &n);
    atom->n_lines   = n;
    printf("  Line columns: %d lines loaded (including A_ul, B_lu, B_ul, nu)\n", n);

    /* LUMINA_AUL_SCALE[N]_*: per-(Z,ion,λ-band) A_ul/f_lu/B_lu/B_ul scale.
     * Pass N="" (primary) and N="2","3" (stacked) for independent per-species rules.
     *   LUMINA_AUL_SCALE[N]_FACTOR    multiplier (default 1.0 = off)
     *   LUMINA_AUL_SCALE[N]_ZMASK     comma-list of Z (default: all)
     *   LUMINA_AUL_SCALE[N]_IONMASK   comma-list of ion stages (default: II)
     *   LUMINA_AUL_SCALE[N]_LAMBDA_MIN  Å, scale lines with λ >= this (default: 0)
     *   LUMINA_AUL_SCALE[N]_LAMBDA_MAX  Å, scale lines with λ < this  (default: 4000)
     */
    {
        const char *suffixes[] = {"", "2", "3", "4", "5", "6", "7", "8", "9"};
        for (int s = 0; s < 9; s++) {
            char name[64];
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_FACTOR", suffixes[s]);
            const char *e_fac = getenv(name);
            if (!e_fac || !(atof(e_fac) > 0.0) || atof(e_fac) == 1.0) continue;
            double fac = atof(e_fac);
            unsigned int zmask = 0;
            unsigned int imask = (1u << 1);
            double lam_min_A = 0.0;
            double lam_max_A = 4000.0;
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_ZMASK", suffixes[s]);
            const char *e_z = getenv(name);
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_IONMASK", suffixes[s]);
            const char *e_ion = getenv(name);
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_LAMBDA_MIN", suffixes[s]);
            const char *e_lmn = getenv(name);
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_LAMBDA_MAX", suffixes[s]);
            const char *e_lm = getenv(name);
            if (e_z) {
                char buf[256]; strncpy(buf, e_z, 255); buf[255]=0;
                for (char *p = strtok(buf, ", "); p; p = strtok(NULL, ", ")) {
                    int z = atoi(p);
                    if (z > 0 && z < 32) zmask |= (1u << z);
                }
            } else {
                for (int z = 1; z < 32; z++) zmask |= (1u << z);
            }
            if (e_ion) {
                char buf[64]; strncpy(buf, e_ion, 63); buf[63]=0;
                imask = 0;
                for (char *p = strtok(buf, ", "); p; p = strtok(NULL, ", ")) {
                    int ii = atoi(p);
                    if (ii >= 0 && ii < 8) imask |= (1u << ii);
                }
            }
            if (e_lmn) lam_min_A = atof(e_lmn);
            if (e_lm) lam_max_A = atof(e_lm);
            int nhit = 0;
            for (int i = 0; i < atom->n_lines; i++) {
                int Z = atom->line_atomic_number[i];
                int ion = atom->line_ion_number[i];
                double lam_A = atom->line_wavelength_cm[i] * 1e8;
                if (lam_A >= lam_min_A && lam_A < lam_max_A &&
                    Z >= 0 && Z < 32 && ion >= 0 && ion < 8 &&
                    (zmask & (1u << Z)) && (imask & (1u << ion))) {
                    atom->line_A_ul[i] *= fac;
                    atom->line_B_lu[i] *= fac;
                    atom->line_B_ul[i] *= fac;
                    atom->line_f_lu[i] *= fac;
                    nhit++;
                }
            }
            printf("  [AUL_SCALE%s] fac=%.3f λ∈[%.0f,%.0f)Å zmask=0x%x imask=0x%x → %d lines scaled\n",
                   suffixes[s], fac, lam_min_A, lam_max_A, zmask, imask, nhit);
        }
    }

    /* --- Level data from levels.csv --- */
    snprintf(path, sizeof(path), "%s/levels.csv", ref_dir);
    atom->level_Z          = read_csv_column_int(path, "atomic_number", &n);
    atom->level_ion        = read_csv_column_int(path, "ion_number", &n);
    atom->level_num        = read_csv_column_int(path, "level_number", &n);
    atom->level_energy_eV  = read_csv_column(path, "energy_eV", &n);
    atom->level_g          = read_csv_column_int(path, "g", &n);
    atom->level_metastable = read_csv_column_int(path, "metastable", &n);
    atom->n_levels = n;
    {
        int n_label = 0;
        atom->level_configuration_sha256 =
            read_csv_column_sha256(path, "configuration", &n_label);
        if (!atom->level_configuration_sha256 || n_label != atom->n_levels) {
            free(atom->level_configuration_sha256);
            atom->level_configuration_sha256 = NULL;
            fprintf(stderr, "[A2-06][WARN] configuration-label hash table "
                    "disabled: source does not match levels.csv (%d != %d)\n",
                    n_label, atom->n_levels);
        } else {
            printf("  [A2-06] configuration-label hash binding active "
                   "(%d levels)\n", n_label);
        }
    }
    /* Super-level index (CMFGEN f_to_s). Optional column: older references
     * lack it -> default to identity (each full level is its own super level)
     * so the NLTE solve reproduces the level-truncation behaviour. */
    atom->level_super = read_csv_column_int(path, "super_level", &n);
    if (atom->level_super == NULL) {
        atom->level_super = (int *)malloc((size_t)atom->n_levels * sizeof(int));
        for (int l = 0; l < atom->n_levels; l++)
            atom->level_super[l] = atom->level_num[l];
        printf("  Levels: %d loaded (no super_level column -> identity)\n",
               atom->n_levels);
    } else {
        int super_active = 0;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_super[l] != atom->level_num[l]) { super_active = 1; break; }
        }
        printf("  Levels: %d loaded (super_level column present%s)\n",
               atom->n_levels,
               super_active ? ", super-levels active" : ", all identity");
    }
    /* ARTIS-style super-level cutoff (LUMINA_SUPER_CUTOFF=K): the CMFGEN f_to_s
     * only collapses the iron-group ions; O II / Co III / etc. stay identity
     * (340->340) so their 250-order Saha-Boltzmann span keeps the rate matrix
     * ill-conditioned. Apply the ARTIS recipe ION_NLEVELS_EXCITED_NLTE=K
     * (nltepop.cc / artisoptions): per ion, levels 0..K-1 are explicit, all
     * higher levels lump into ONE super-level K (Boltzmann-distributed by the
     * existing within_sl_frac). level_num is the per-ion 0-based energy rank, so
     * super = min(level_num, K). Overrides the loaded f_to_s uniformly (A/B with
     * K=0 = keep loaded). Activate together with LUMINA_SUPER_LEVELS=1. */
    {
        const char *e = getenv("LUMINA_SUPER_CUTOFF");
        int K = e ? atoi(e) : 0;
        if (K > 0) {
            long ncollapsed = 0;
            for (int l = 0; l < atom->n_levels; l++) {
                if (atom->level_num[l] >= K) { atom->level_super[l] = K; ncollapsed++; }
                else                          atom->level_super[l] = atom->level_num[l];
            }
            printf("  [ARTIS super-cutoff] K=%d: %ld levels lumped → per-ion ≤%d explicit + 1 super-level "
                   "(set LUMINA_SUPER_LEVELS=1 to activate)\n", K, ncollapsed, K);
        }
    }

    /* --- Ionization energies --- */
    snprintf(path, sizeof(path), "%s/ionization_energies.csv", ref_dir);
    atom->ioniz_Z         = read_csv_column_int(path, "atomic_number", &n);
    atom->ioniz_ion       = read_csv_column_int(path, "ion_number", &n);
    atom->ioniz_energy_eV = read_csv_column(path, "ionization_energy_eV", &n);
    atom->n_ionization = n;
    printf("  Ionization: %d entries\n", n);

    /* --- Zeta data --- */
    snprintf(path, sizeof(path), "%s/zeta_ions.csv", ref_dir);
    atom->zeta_Z   = read_csv_column_int(path, "atomic_number", &n);
    atom->zeta_ion = read_csv_column_int(path, "ion_number", &n);
    atom->n_zeta_ions = n;

    snprintf(path, sizeof(path), "%s/zeta_temps.csv", ref_dir);
    atom->zeta_temps = read_csv_column(path, "temperature", &n);
    atom->n_zeta_temps = n;

    snprintf(path, sizeof(path), "%s/zeta_data.npy", ref_dir);
    int zr, zc;
    atom->zeta_data = read_npy_f64(path, &zr, &zc);
    printf("  Zeta: %d ions x %d temps, data [%d x %d]\n",
           atom->n_zeta_ions, atom->n_zeta_temps, zr, zc);

    /* --- Atom masses --- */
    snprintf(path, sizeof(path), "%s/atom_masses.csv", ref_dir);
    atom->element_Z        = read_csv_column_int(path, "atomic_number", &n);
    atom->element_mass_amu = read_csv_column(path, "mass_amu", &n);
    atom->n_elements = n;
    printf("  Elements: %d (", n);
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", atom->element_Z[i]);
    printf(")\n");

    /* --- Abundances --- */
    snprintf(path, sizeof(path), "%s/abundances.csv", ref_dir);
    atom->abundances = (double *)calloc((size_t)atom->n_elements * n_shells,
                                        sizeof(double));
    {
        /* Composition CSV contract (ORDER_CD_COMPOSITION_IDENTITY.md, order D).
         * Read physical lines byte-by-byte so an embedded NUL cannot masquerade
         * as end-of-string and an overlong line cannot be accepted in chunks. */
        FILE *mass_fp = NULL;
        FILE *ab_fp = fopen(path, "rb");
        char line[8192];
        int *mass_seen_z = NULL;
        unsigned char *ab_seen = NULL;
        double *shell_sum = NULL;
        double *row_values = NULL;
        int fatal_seen = 0;

        #define D_IS_SPACE(c) \
            ((c) == ' ' || (c) == '\t' || (c) == '\n' || (c) == '\r' || \
             (c) == '\v' || (c) == '\f')
        #define D_READ_LINE(stream, buffer, length, got, nul, overlong) \
            do { \
                int d_ch; \
                (length) = 0; (got) = 0; (nul) = 0; (overlong) = 0; \
                while ((d_ch = fgetc((stream))) != EOF) { \
                    (got) = 1; \
                    if (d_ch == 0) (nul) = 1; \
                    if ((length) + 1 < sizeof(buffer)) \
                        (buffer)[(length)++] = (char)d_ch; \
                    else \
                        (overlong) = 1; \
                    if (d_ch == '\n') break; \
                } \
                (buffer)[(length)] = '\0'; \
            } while (0)
        #define D_CLEANUP() \
            do { \
                if (mass_fp) fclose(mass_fp); \
                if (ab_fp) fclose(ab_fp); \
                free(mass_seen_z); free(ab_seen); free(shell_sum); \
                free(row_values); \
            } while (0)
        #define D_RETURN_FATAL(...) \
            do { \
                fprintf(stderr, __VA_ARGS__); \
                D_CLEANUP(); \
                return 1; \
            } while (0)

        if (!ab_fp) {
            D_RETURN_FATAL("[D1][FATAL] cannot open %s\n", path);
        }
        if (!atom->abundances || atom->n_elements <= 0 || n_shells <= 0) {
            D_RETURN_FATAL("[D14][FATAL] invalid atom-mass row count (%d) or "
                           "shell count (%d)\n", atom->n_elements, n_shells);
        }

        /* Re-read atom_masses.csv strictly.  The legacy column loaders above
         * construct the arrays; this pass validates that both columns describe
         * the same positive, finite, unique set without changing their values. */
        {
            char mass_path[512];
            size_t len;
            int got, had_nul, overlong;
            int mass_rows = 0, mass_valid_rows = 0;
            int mass_non_d9_fatal = 0;
            int mass_seen_count = 0;
            int mass_seen_capacity = atom->n_elements > 0 ? atom->n_elements : 1;

            snprintf(mass_path, sizeof(mass_path), "%s/atom_masses.csv", ref_dir);
            mass_fp = fopen(mass_path, "rb");
            if (!mass_fp) {
                D_RETURN_FATAL("[D14][FATAL] cannot open %s\n", mass_path);
            }
            mass_seen_z = (int *)malloc((size_t)mass_seen_capacity * sizeof(int));
            if (!mass_seen_z) {
                D_RETURN_FATAL("[D14][FATAL] out of memory validating %s\n",
                               mass_path);
            }

            D_READ_LINE(mass_fp, line, len, got, had_nul, overlong);
            if (overlong) {
                D_RETURN_FATAL("[D10][FATAL] %s header exceeds %zu-byte line "
                               "buffer\n", mass_path, sizeof(line));
            }
            if (!got) {
                D_RETURN_FATAL("[D14][FATAL] %s has no header\n", mass_path);
            }
            if (had_nul) {
                D_RETURN_FATAL("[D8][FATAL] %s header contains a NUL byte\n",
                               mass_path);
            }
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';
            if (strcmp(line, "atomic_number,mass_amu") != 0) {
                D_RETURN_FATAL("[D14][FATAL] %s header must be "
                               "atomic_number,mass_amu\n", mass_path);
            }

            for (;;) {
                char *comma, *z_end, *mass_end;
                char *q;
                long z_value = 0;
                double mass_value = 0.0;
                int row_bad = 0;

                D_READ_LINE(mass_fp, line, len, got, had_nul, overlong);
                if (!got) break;
                mass_rows++;
                if (overlong) {
                    D_RETURN_FATAL("[D10][FATAL] %s row %d exceeds %zu-byte "
                                   "line buffer\n", mass_path, mass_rows + 1,
                                   sizeof(line));
                }
                if (had_nul) {
                    fprintf(stderr, "[D8][FATAL] %s row %d contains a NUL byte\n",
                            mass_path, mass_rows + 1);
                    mass_non_d9_fatal = 1;
                    continue;
                }
                while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                    line[--len] = '\0';

                comma = strchr(line, ',');
                if (!comma || strchr(comma + 1, ',')) {
                    fprintf(stderr, "[D14][FATAL] %s row %d must contain "
                            "exactly two fields\n", mass_path, mass_rows + 1);
                    mass_non_d9_fatal = 1;
                    continue;
                }

                errno = 0;
                z_value = strtol(line, &z_end, 10);
                if (errno == ERANGE) {
                    fprintf(stderr, "[D17][FATAL] %s row %d Z is outside strtol "
                            "range\n", mass_path, mass_rows + 1);
                    mass_non_d9_fatal = 1;
                    row_bad = 1;
                }
                q = z_end;
                while (q < comma && D_IS_SPACE((unsigned char)*q)) q++;
                if (z_end == line || q != comma || z_value <= 0 ||
                    z_value > INT_MAX) {
                    fprintf(stderr, "[D14][FATAL] %s row %d Z must be a positive "
                            "integer representable as int\n", mass_path,
                            mass_rows + 1);
                    mass_non_d9_fatal = 1;
                    row_bad = 1;
                }

                errno = 0;
                mass_value = strtod(comma + 1, &mass_end);
                if (errno == ERANGE) {
                    fprintf(stderr, "[D17][FATAL] %s row %d mass is outside "
                            "strtod range\n", mass_path, mass_rows + 1);
                    mass_non_d9_fatal = 1;
                    row_bad = 1;
                }
                q = mass_end;
                while (*q && D_IS_SPACE((unsigned char)*q)) q++;
                if (mass_end == comma + 1 || *q != '\0' ||
                    !(mass_value > 0.0) || !isfinite(mass_value)) {
                    fprintf(stderr, "[D14][FATAL] %s row %d mass must be a "
                            "positive finite number with no trailing garbage\n",
                            mass_path, mass_rows + 1);
                    mass_non_d9_fatal = 1;
                    row_bad = 1;
                }
                if (row_bad) continue;

                for (int i = 0; i < mass_seen_count; i++) {
                    if (mass_seen_z[i] == (int)z_value) {
                        fprintf(stderr, "[D9][FATAL] duplicate Z=%ld in %s\n",
                                z_value, mass_path);
                        fatal_seen = 1;
                        break;
                    }
                }
                if (mass_seen_count == mass_seen_capacity) {
                    int new_capacity = mass_seen_capacity * 2;
                    int *grown = (int *)realloc(
                        mass_seen_z, (size_t)new_capacity * sizeof(int));
                    if (!grown) {
                        D_RETURN_FATAL("[D14][FATAL] out of memory validating "
                                       "%s\n", mass_path);
                    }
                    mass_seen_z = grown;
                    mass_seen_capacity = new_capacity;
                }
                mass_seen_z[mass_seen_count++] = (int)z_value;
                mass_valid_rows++;
            }
            if (ferror(mass_fp)) {
                D_RETURN_FATAL("[D14][FATAL] read error in %s\n", mass_path);
            }
            fclose(mass_fp);
            mass_fp = NULL;

            if (mass_rows == 0 || mass_rows != atom->n_elements ||
                mass_valid_rows != mass_rows) {
                fprintf(stderr, "[D14][FATAL] %s row-count mismatch: physical=%d, "
                        "valid=%d, loaded=%d\n", mass_path, mass_rows,
                        mass_valid_rows, atom->n_elements);
                mass_non_d9_fatal = 1;
            }
            if (mass_non_d9_fatal) {
                D_CLEANUP();
                return 1;
            }
        }

        ab_seen = (unsigned char *)calloc((size_t)atom->n_elements, 1);
        shell_sum = (double *)calloc((size_t)n_shells, sizeof(double));
        row_values = (double *)malloc((size_t)n_shells * sizeof(double));
        if (!ab_seen || !shell_sum || !row_values) {
            D_RETURN_FATAL("[D14][FATAL] out of memory validating composition\n");
        }

        /* Header contract: exactly atomic_number,0,1,...,n_shells-1. */
        {
            size_t len;
            int got, had_nul, overlong;
            int total_fields = 1;
            int schema_bad = 0;
            char *field;

            D_READ_LINE(ab_fp, line, len, got, had_nul, overlong);
            if (overlong) {
                D_RETURN_FATAL("[D10][FATAL] %s header exceeds %zu-byte line "
                               "buffer\n", path, sizeof(line));
            }
            if (!got) {
                D_RETURN_FATAL("[D2][FATAL] %s has no header\n", path);
            }
            if (had_nul) {
                D_RETURN_FATAL("[D8][FATAL] %s header contains a NUL byte\n", path);
            }
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';
            for (size_t i = 0; i < len; i++)
                if (line[i] == ',') total_fields++;
            if (total_fields - 1 != n_shells) {
                D_RETURN_FATAL("[D2][FATAL] %s header has %d shell columns; "
                               "expected %d\n", path, total_fields - 1,
                               n_shells);
            }

            field = line;
            for (int col = 0; col < total_fields; col++) {
                char *end = strchr(field, ',');
                char expected[32];
                size_t actual_len;
                if (!end) end = field + strlen(field);
                actual_len = (size_t)(end - field);
                if (col == 0)
                    snprintf(expected, sizeof(expected), "atomic_number");
                else
                    snprintf(expected, sizeof(expected), "%d", col - 1);
                if (strlen(expected) != actual_len ||
                    strncmp(field, expected, actual_len) != 0)
                    schema_bad = 1;
                field = *end == ',' ? end + 1 : end;
            }
            if (schema_bad) {
                D_RETURN_FATAL("[D12][FATAL] %s header schema/order must be "
                               "atomic_number,0,1,...,%d\n", path,
                               n_shells - 1);
            }
        }

        /* Read to EOF, including rows beyond atom->n_elements. */
        {
            int physical_row = 1;
            int recognized_rows = 0;
            for (;;) {
                size_t len;
                int got, had_nul, overlong;
                int shell_fields = 0;
                int eidx = -1;
                int row_bad = 0;
                char *p, *end, *q;
                long z_value;

                D_READ_LINE(ab_fp, line, len, got, had_nul, overlong);
                if (!got) break;
                physical_row++;
                if (overlong) {
                    D_RETURN_FATAL("[D10][FATAL] %s row %d exceeds %zu-byte line "
                                   "buffer\n", path, physical_row,
                                   sizeof(line));
                }
                if (had_nul) {
                    fprintf(stderr, "[D8][FATAL] %s row %d contains a NUL byte\n",
                            path, physical_row);
                    fatal_seen = 1;
                    continue;
                }
                while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                    line[--len] = '\0';
                for (size_t i = 0; i < len; i++)
                    if (line[i] == ',') shell_fields++;
                if (shell_fields != n_shells) {
                    fprintf(stderr, "[D3][FATAL] %s row %d has %d shell fields; "
                            "expected %d\n", path, physical_row, shell_fields,
                            n_shells);
                    fatal_seen = 1;
                    continue;
                }

                p = line;
                errno = 0;
                z_value = strtol(p, &end, 10);
                if (errno == ERANGE) {
                    fprintf(stderr, "[D17][FATAL] %s row %d Z is outside strtol "
                            "range\n", path, physical_row);
                    fatal_seen = 1;
                    continue;
                }
                q = end;
                while (D_IS_SPACE((unsigned char)*q)) q++;
                if (end == p || *q != ',') {
                    fprintf(stderr, "[D8][FATAL] %s row %d has an invalid Z token "
                            "or Z-token trailing garbage\n", path, physical_row);
                    fatal_seen = 1;
                    continue;
                }
                if (z_value <= 0 || z_value > INT_MAX) {
                    fprintf(stderr, "[D4][FATAL] %s row %d Z=%ld is absent from "
                            "atom_masses.csv\n", path, physical_row, z_value);
                    fatal_seen = 1;
                    continue;
                }
                for (int i = 0; i < atom->n_elements; i++) {
                    if (atom->element_Z[i] == (int)z_value) {
                        eidx = i;
                        break;
                    }
                }
                if (eidx < 0) {
                    fprintf(stderr, "[D4][FATAL] %s row %d Z=%ld is absent from "
                            "atom_masses.csv\n", path, physical_row, z_value);
                    fatal_seen = 1;
                    continue;
                }
                if (ab_seen[eidx]) {
                    fprintf(stderr, "[D9][FATAL] duplicate Z=%ld in %s\n",
                            z_value, path);
                    fatal_seen = 1;
                } else {
                    ab_seen[eidx] = 1;
                }

                p = q + 1;
                for (int s = 0; s < n_shells; s++) {
                    double value;
                    errno = 0;
                    value = strtod(p, &end);
                    if (errno == ERANGE) {
                        fprintf(stderr, "[D17][FATAL] %s row %d shell %d is "
                                "outside strtod range\n", path, physical_row, s);
                        fatal_seen = 1;
                        row_bad = 1;
                    }
                    q = end;
                    while (D_IS_SPACE((unsigned char)*q)) q++;
                    if (end == p || (s + 1 < n_shells ? *q != ',' : *q != '\0')) {
                        fprintf(stderr, "[D8][FATAL] %s row %d shell %d has an "
                                "invalid number or trailing garbage\n", path,
                                physical_row, s);
                        fatal_seen = 1;
                        row_bad = 1;
                    }
                    if (isnan(value)) {
                        fprintf(stderr, "[D7a][FATAL] %s row %d shell %d is NaN\n",
                                path, physical_row, s);
                        fatal_seen = 1;
                        row_bad = 1;
                    } else if (!isfinite(value)) {
                        fprintf(stderr, "[D7b][FATAL] %s row %d shell %d is "
                                "infinite\n", path, physical_row, s);
                        fatal_seen = 1;
                        row_bad = 1;
                    } else if (value < 0.0) {
                        fprintf(stderr, "[D7c][FATAL] %s row %d shell %d is "
                                "negative (%.17g)\n", path, physical_row, s,
                                value);
                        fatal_seen = 1;
                        row_bad = 1;
                    } else if (value > 1.0) {
                        fprintf(stderr, "[D16][FATAL] %s row %d shell %d exceeds "
                                "one (%.17g)\n", path, physical_row, s, value);
                        fatal_seen = 1;
                        row_bad = 1;
                    }
                    row_values[s] = value;
                    p = *q == ',' ? q + 1 : q;
                }
                if (row_bad) continue;

                for (int s = 0; s < n_shells; s++) {
                    atom->abundances[eidx * n_shells + s] = row_values[s];
                    shell_sum[s] += row_values[s];
                }
                recognized_rows++;
            }
            if (ferror(ab_fp)) {
                D_RETURN_FATAL("[D8][FATAL] read error in %s\n", path);
            }
            fclose(ab_fp);
            ab_fp = NULL;

            if (fatal_seen) {
                D_CLEANUP();
                return 1;
            }
            if (recognized_rows == 0) {
                D_RETURN_FATAL("[D13][FATAL] %s contains zero recognized data "
                               "rows\n", path);
            }
        }

        for (int s = 0; s < n_shells; s++) {
            if (!(shell_sum[s] > 0.0)) {
                D_RETURN_FATAL("[D15][FATAL] shell %d abundance sum is not "
                               "positive (%.17g)\n", s, shell_sum[s]);
            }
        }

        /* Warnings are intentionally observable on stdout. */
        {
            int missing = 0;
            for (int i = 0; i < atom->n_elements; i++)
                if (!ab_seen[i]) missing++;
            if (missing > 0) {
                int emitted = 0;
                printf("[D5][WARN] %d atom_masses.csv element(s) absent from "
                       "abundances.csv\n", missing);
                printf("  missing Z: ");
                for (int i = 0; i < atom->n_elements; i++) {
                    if (!ab_seen[i]) {
                        printf("%s%d", emitted ? "," : "", atom->element_Z[i]);
                        emitted++;
                    }
                }
                printf("\n");
            }
        }
        {
            int bad_sums = 0;
            for (int s = 0; s < n_shells; s++)
                if (fabs(shell_sum[s] - 1.0) > 1e-6) bad_sums++;
            if (bad_sums > 0) {
                int emitted = 0;
                printf("[D6][WARN] %d shell abundance sum(s) differ from one by "
                       "more than 1e-6\n", bad_sums);
                printf("  shell sums: ");
                for (int s = 0; s < n_shells; s++) {
                    if (fabs(shell_sum[s] - 1.0) > 1e-6) {
                        printf("%s%d=%.17g", emitted ? "," : "", s,
                               shell_sum[s]);
                        emitted++;
                    }
                }
                printf("\n");
            }
        }

        D_CLEANUP();
        #undef D_RETURN_FATAL
        #undef D_CLEANUP
        #undef D_READ_LINE
        #undef D_IS_SPACE
    }

    /* --- Build ion population table --- */
    /* For each element, ion stages go from 0 to n_ionization_entries_for_element */
    /* Count total ion populations */
    atom->elem_ion_offset = (int *)calloc(atom->n_elements + 1, sizeof(int));
    int total_ion_pops = 0;
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        int n_ioniz = 0;
        for (int i = 0; i < atom->n_ionization; i++) {
            if (atom->ioniz_Z[i] == z) n_ioniz++;
        }
        atom->elem_ion_offset[e] = total_ion_pops;
        total_ion_pops += n_ioniz + 1; /* n_ioniz energies -> n_ioniz+1 populations */
    }
    atom->elem_ion_offset[atom->n_elements] = total_ion_pops;
    atom->n_ion_pops = total_ion_pops;

    atom->ion_pop_Z     = (int *)calloc(total_ion_pops, sizeof(int));
    atom->ion_pop_stage = (int *)calloc(total_ion_pops, sizeof(int));
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        int n_pops = atom->elem_ion_offset[e + 1] - atom->elem_ion_offset[e];
        /* The ion ladder starts at the LOWEST stage that has atomic data, not
           necessarily neutral. CMFGEN omits neutral Ti I / Mn I, so their
           ionization-energy entries start at stage 1; labelling pops relatively
           (k) dumped all mass into a phantom level-less neutral slot. Anchor the
           ladder at base_stage = min ionization-energy stage for the element. */
        int base_stage = 0;
        int found_base = 0;
        for (int i = 0; i < atom->n_ionization; i++) {
            if (atom->ioniz_Z[i] == z &&
                (!found_base || atom->ioniz_ion[i] < base_stage)) {
                base_stage = atom->ioniz_ion[i];
                found_base = 1;
            }
        }
        /* found_base==0 (no ioniz data): leave base_stage=0 (assume neutral) */
        for (int k = 0; k < n_pops; k++) {
            int idx = atom->elem_ion_offset[e] + k;
            atom->ion_pop_Z[idx] = z;
            atom->ion_pop_stage[idx] = base_stage + k;
        }
    }
    printf("  Ion populations: %d total\n", total_ion_pops);

    /* --- Build level lookup: level_offset[ion_pop_idx] --- */
    /* Levels are sorted by (Z, ion, level_num) in levels.csv */
    atom->level_offset = (int *)calloc(total_ion_pops + 1, sizeof(int));
    for (int ip = 0; ip < total_ion_pops; ip++) {
        int z = atom->ion_pop_Z[ip];
        int ion_stage = atom->ion_pop_stage[ip];
        int count = 0;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_Z[l] == z && atom->level_ion[l] == ion_stage) count++;
        }
        atom->level_offset[ip + 1] = atom->level_offset[ip] + count;
    }
    printf("  Level offsets built: %d total levels mapped\n",
           atom->level_offset[total_ion_pops]);

    /* Verify level_offset total matches n_levels */
    if (atom->level_offset[total_ion_pops] != atom->n_levels) {
        fprintf(stderr, "WARNING: level_offset total %d != n_levels %d\n",
                atom->level_offset[total_ion_pops], atom->n_levels);
    }

    /* --- [ALPHA-SPINGATE] optional per-level spin multiplicity (2S+1) ---------
     * Loaded ONLY when LUMINA_ALPHA_SPINGATE=1 -> when the gate is off this
     * whole block is skipped and atom->level_mult stays NULL (memset above),
     * so the OFF-path heap layout is byte-for-byte unchanged. The companion
     * (level_multiplicity.csv) is produced offline from CMFGEN OSC term labels
     * (scripts/bake_level_multiplicity.py) and keyed by (atomic_number,
     * ion_number, level_number) so it is portable across reference dirs that
     * share the CMFGEN level ordering. Levels absent from the companion stay 0
     * (unknown -> never skipped downstream). */
    atom->level_mult = NULL;
    {
        const char *sg = getenv("LUMINA_ALPHA_SPINGATE");
        if (sg && atoi(sg)) {
            char mpath[512];
            const char *ovr = getenv("LUMINA_SPINGATE_MULT");
            if (ovr && ovr[0])
                snprintf(mpath, sizeof(mpath), "%s", ovr);
            else
                snprintf(mpath, sizeof(mpath), "%s/level_multiplicity.csv", ref_dir);
            FILE *probe = fopen(mpath, "r");
            if (!probe) {
                fprintf(stderr, "  [ALPHA-SPINGATE][WARN] multiplicity table not "
                        "found at %s -> gate will be inert (no levels skipped)\n",
                        mpath);
            } else {
                fclose(probe);
                int nz = 0, ni = 0, nl = 0, nm = 0;
                int *m_Z    = read_csv_column_int(mpath, "atomic_number", &nz);
                int *m_ion  = read_csv_column_int(mpath, "ion_number",    &ni);
                int *m_lnum = read_csv_column_int(mpath, "level_number",  &nl);
                int *m_mult = read_csv_column_int(mpath, "multiplicity",  &nm);
                if (m_Z && m_ion && m_lnum && m_mult &&
                    nz == ni && ni == nl && nl == nm) {
                    atom->level_mult = (signed char *)calloc((size_t)atom->n_levels,
                                                             sizeof(signed char));
                    long filled = 0;
                    for (int r = 0; r < nz; r++) {
                        int mm = m_mult[r];
                        if (mm < 0) mm = 0; else if (mm > 127) mm = 127;
                        if (mm == 0) continue;
                        /* find ion-pop for (Z, ion_charge) */
                        int ip = -1;
                        for (int q = 0; q < total_ion_pops; q++) {
                            if (atom->ion_pop_Z[q] == m_Z[r] &&
                                atom->ion_pop_stage[q] == m_ion[r]) { ip = q; break; }
                        }
                        if (ip < 0) continue;
                        int gi = atom->level_offset[ip] + m_lnum[r];
                        if (gi < atom->level_offset[ip] ||
                            gi >= atom->level_offset[ip + 1]) continue;
                        /* guard against ordering drift: verify identity */
                        if (atom->level_Z[gi] != m_Z[r] ||
                            atom->level_ion[gi] != m_ion[r] ||
                            atom->level_num[gi] != m_lnum[r]) continue;
                        atom->level_mult[gi] = (signed char)mm;
                        filled++;
                    }
                    printf("  [ALPHA-SPINGATE] multiplicity table: %ld/%d levels "
                           "assigned (%.1f%%) from %s\n", filled, atom->n_levels,
                           100.0 * (double)filled / (double)atom->n_levels, mpath);
                } else {
                    fprintf(stderr, "  [ALPHA-SPINGATE][WARN] malformed table %s "
                            "(cols %d/%d/%d/%d) -> gate inert\n",
                            mpath, nz, ni, nl, nm);
                }
                free(m_Z); free(m_ion); free(m_lnum); free(m_mult);
            }
        }
    }

    /* --- Allocate per-shell computed arrays --- */
    atom->ion_number_density  = (double *)calloc((size_t)total_ion_pops * n_shells, sizeof(double));
    atom->partition_functions = (double *)calloc((size_t)total_ion_pops * n_shells, sizeof(double));

    /* The bf repairs own their target-map dependency: neither stimulated
     * recombination nor the D-1 event selector may require MA_RADRECOMB. */
    {
        const char *stim = getenv("LUMINA_FIX_BF_STIM_RECOMB");
        const char *event = getenv("LUMINA_FIX_BF_CONTINUUM_EVENT");
        if ((stim && atoi(stim) != 0) || (event && atoi(event) != 0) ||
            nlte_element_wide_enabled()) {
            const char *override = getenv("LUMINA_MA_RADRECOMB_TARGET");
            char target_path[1024];
            if (override && strchr(override, '/'))
                snprintf(target_path, sizeof(target_path), "%s", override);
            else
                snprintf(target_path, sizeof(target_path),
                         "%s/ma_radrecomb_target.bin", ref_dir);
            load_ma_radrecomb_target(atom, target_path);
        }
    }

    printf("Atomic data loading complete.\n");
    return 0;
}

/* Load pre-baked CMFGEN sigma_bf grid (cmfgen_sigma_bf.bin).
 * Binary layout (little-endian, written by scripts/expand_atomic_data_cmfgen.py):
 *   uint32 magic   = 0x434D4644 ('CMFD')
 *   uint32 version = 1
 *   int32  n_levels, n_freq_bins
 *   double nu_min_Hz, nu_max_Hz
 *   int8   has_cmfgen[n_levels]   (padded to 8-byte alignment)
 *   double sigma_cm2[n_levels * n_freq_bins]
 *
 * Returns 0 on success (grid loaded into atom->cmfgen_*), -1 on missing file
 * or schema mismatch (atom->cmfgen_loaded stays 0 → Kramers fallback). */
int load_cmfgen_sigma_bf(AtomicData *atom, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  CMFGEN sigma_bf: %s not found, using Kramers fallback\n", path);
        atom->cmfgen_loaded = 0;
        return -1;
    }
    uint32_t magic = 0, version = 0;
    int32_t n_lev = 0, n_freq = 0;
    double nu_min = 0.0, nu_max = 0.0;
    if (fread(&magic, 4, 1, fp) != 1 || fread(&version, 4, 1, fp) != 1 ||
        fread(&n_lev, 4, 1, fp) != 1 || fread(&n_freq, 4, 1, fp) != 1 ||
        fread(&nu_min, 8, 1, fp) != 1 || fread(&nu_max, 8, 1, fp) != 1) {
        fprintf(stderr, "ERROR: %s header read failed\n", path);
        fclose(fp);
        return -1;
    }
    if (magic != 0x434D4644u || version != 1u) {
        fprintf(stderr, "ERROR: %s bad magic/version (0x%08x v%u)\n",
                path, magic, version);
        fclose(fp);
        return -1;
    }
    if (n_lev != atom->n_levels) {
        fprintf(stderr, "ERROR: %s n_levels=%d != atom->n_levels=%d\n",
                path, n_lev, atom->n_levels);
        fclose(fp);
        return -1;
    }
    if (n_freq != NLTE_N_FREQ_BINS) {
        fprintf(stderr, "ERROR: %s n_freq_bins=%d != NLTE_N_FREQ_BINS=%d\n",
                path, n_freq, NLTE_N_FREQ_BINS);
        fclose(fp);
        return -1;
    }

    /* Read has_cmfgen[n_levels] as int8 then promote to int */
    int8_t *flag8 = (int8_t *)malloc((size_t)n_lev);
    if (fread(flag8, 1, (size_t)n_lev, fp) != (size_t)n_lev) {
        fprintf(stderr, "ERROR: %s has_cmfgen read failed\n", path);
        free(flag8);
        fclose(fp);
        return -1;
    }
    /* Skip 8-byte alignment pad */
    int pad = (8 - (n_lev % 8)) % 8;
    if (pad > 0) fseek(fp, pad, SEEK_CUR);

    atom->cmfgen_has_sigma = (int *)malloc((size_t)n_lev * sizeof(int));
    int n_with = 0;
    for (int i = 0; i < n_lev; i++) {
        atom->cmfgen_has_sigma[i] = flag8[i];
        if (flag8[i]) n_with++;
    }
    free(flag8);

    size_t grid_n = (size_t)n_lev * (size_t)n_freq;
    atom->cmfgen_sigma_bf = (double *)malloc(grid_n * sizeof(double));
    if (fread(atom->cmfgen_sigma_bf, sizeof(double), grid_n, fp) != grid_n) {
        fprintf(stderr, "ERROR: %s sigma grid read failed\n", path);
        free(atom->cmfgen_has_sigma);
        free(atom->cmfgen_sigma_bf);
        atom->cmfgen_has_sigma = NULL;
        atom->cmfgen_sigma_bf = NULL;
        fclose(fp);
        return -1;
    }
    fclose(fp);

    atom->cmfgen_n_freq_bins = n_freq;
    atom->cmfgen_nu_min = nu_min;
    atom->cmfgen_nu_max = nu_max;
    atom->cmfgen_loaded = 1;
    printf("  CMFGEN sigma_bf: %d/%d levels (%.1f%%) on %d-bin grid\n",
           n_with, n_lev, 100.0 * n_with / (double)n_lev, n_freq);
    return 0;
}

/* Load the bound-free upper-ion photoionization TARGET map (B4/D1/M1 data gap).
 * Parallel to cmfgen_sigma_bf.bin (same n_levels ordering); records per level the
 * GLOBAL index of the upper-ion level a photoionization lands on (== the recomb
 * source). Built by scripts/build_ma_radrecomb_target.py. Layout (little-endian):
 *   uint32 magic   = 0x4D415254 ('MART')
 *   uint32 version = 1 or 2
 *   int32  n_levels                (must == atom->n_levels)
 *   int32  n_ions_mapped
 * v1:
 *   int32  target_level_idx[n_levels]   (upper-ion target global level, or -1)
 * v2:
 *   int32  n_routes
 *   int32  target_offset[n_levels+1]
 *   int32  target_level_idx[n_routes]
 *   double target_probability[n_routes]
 * Version 1 is the established CMFGEN single-target schema and therefore has
 * one p=1 route for every mapped level. Version 2 is a general multi-target CSR.
 * Returns 0 on success (atom->ma_rr_loaded=1); -1 on any problem (stays 0 => the
 * MA-RADRECOMB gate degrades to the ground-only assumption / no continuum). */
int load_ma_radrecomb_target(AtomicData *atom, const char *path) {
    atom->ma_rr_loaded = 0;
    atom->ma_rr_target = NULL;
    atom->ma_rr_n_routes = 0;
    atom->ma_rr_target_offset = NULL;
    atom->ma_rr_targets = NULL;
    atom->ma_rr_probability = NULL;
    atom->ma_rr_n_ions = 0;
    atom->ma_rr_n_mapped = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  [MA-RADRECOMB] target map %s not found -> ground-only fallback\n",
               path);
        return -1;
    }
    uint32_t magic = 0, version = 0;
    int32_t n_lev = 0, n_ions = 0;
    if (fread(&magic, 4, 1, fp) != 1 || fread(&version, 4, 1, fp) != 1 ||
        fread(&n_lev, 4, 1, fp) != 1 || fread(&n_ions, 4, 1, fp) != 1) {
        fprintf(stderr, "ERROR: %s header read failed\n", path);
        fclose(fp); return -1;
    }
    if (magic != 0x4D415254u || (version != 1u && version != 2u)) {
        fprintf(stderr, "ERROR: %s bad magic/version (0x%08x v%u)\n",
                path, magic, version);
        fclose(fp); return -1;
    }
    if (n_lev != atom->n_levels) {
        fprintf(stderr, "ERROR: %s n_levels=%d != atom->n_levels=%d\n",
                path, n_lev, atom->n_levels);
        fclose(fp); return -1;
    }
    int *primary = (int *)malloc((size_t)n_lev * sizeof(int));
    int *offset = (int *)malloc((size_t)(n_lev + 1) * sizeof(int));
    if (!primary || !offset) {
        free(primary); free(offset); fclose(fp); return -1;
    }
    for (int i = 0; i < n_lev; i++) primary[i] = -1;

    int32_t n_routes32 = 0;
    int *targets = NULL;
    double *prob = NULL;
    if (version == 1u) {
        int *legacy = (int *)malloc((size_t)n_lev * sizeof(int));
        if (!legacy ||
            fread(legacy, sizeof(int), (size_t)n_lev, fp) != (size_t)n_lev) {
            fprintf(stderr, "ERROR: %s v1 target array read failed\n", path);
            free(legacy); free(primary); free(offset); fclose(fp); return -1;
        }
        for (int i = 0; i < n_lev; i++)
            if (legacy[i] >= 0 && legacy[i] < n_lev) n_routes32++;
        size_t nr_alloc = n_routes32 > 0 ? (size_t)n_routes32 : 1u;
        targets = (int *)malloc(nr_alloc * sizeof(int));
        prob = (double *)malloc(nr_alloc * sizeof(double));
        if ((n_routes32 > 0) && (!targets || !prob)) {
            free(legacy); free(primary); free(offset);
            free(targets); free(prob); fclose(fp); return -1;
        }
        int r = 0;
        offset[0] = 0;
        for (int i = 0; i < n_lev; i++) {
            if (legacy[i] >= 0 && legacy[i] < n_lev) {
                primary[i] = legacy[i];
                targets[r] = legacy[i];
                prob[r] = 1.0;
                r++;
            }
            offset[i + 1] = r;
        }
        free(legacy);
    } else {
        if (fread(&n_routes32, sizeof(int32_t), 1, fp) != 1 ||
            n_routes32 < 0 ||
            fread(offset, sizeof(int), (size_t)n_lev + 1, fp) !=
                (size_t)n_lev + 1) {
            fprintf(stderr, "ERROR: %s v2 CSR header/offset read failed\n", path);
            free(primary); free(offset); fclose(fp); return -1;
        }
        size_t nr_alloc = n_routes32 > 0 ? (size_t)n_routes32 : 1u;
        targets = (int *)malloc(nr_alloc * sizeof(int));
        prob = (double *)malloc(nr_alloc * sizeof(double));
        if ((n_routes32 > 0) && (!targets || !prob)) {
            free(primary); free(offset); free(targets); free(prob);
            fclose(fp); return -1;
        }
        if (fread(targets, sizeof(int), (size_t)n_routes32, fp) !=
                (size_t)n_routes32 ||
            fread(prob, sizeof(double), (size_t)n_routes32, fp) !=
                (size_t)n_routes32) {
            fprintf(stderr, "ERROR: %s v2 route arrays read failed\n", path);
            free(primary); free(offset); free(targets); free(prob);
            fclose(fp); return -1;
        }
    }
    fclose(fp);

    /* Fail closed on malformed CSR or invalid targets/probabilities. Unlike a
     * single primary map, route compaction preserves every valid target. */
    int bad_csr = offset[0] != 0 || offset[n_lev] != n_routes32;
    for (int i = 0; i < n_lev && !bad_csr; i++)
        if (offset[i] > offset[i + 1] || offset[i] < 0 ||
            offset[i + 1] > n_routes32) bad_csr = 1;
    int n_mapped = 0, n_scrub = 0;
    if (!bad_csr) {
        for (int i = 0; i < n_lev; i++) {
            int first_valid = -1;
            for (int r = offset[i]; r < offset[i + 1]; r++) {
                if (targets[r] < 0 || targets[r] >= n_lev ||
                    !isfinite(prob[r]) || prob[r] < 0.0 || prob[r] > 1.0) {
                    prob[r] = 0.0;
                    n_scrub++;
                    continue;
                }
                if (first_valid < 0 && prob[r] > 0.0) first_valid = targets[r];
            }
            primary[i] = first_valid;
            if (first_valid >= 0) n_mapped++;
        }
    } else {
        fprintf(stderr, "ERROR: %s malformed v2 target CSR\n", path);
        free(primary); free(offset); free(targets); free(prob);
        return -1;
    }
    atom->ma_rr_target = primary;
    atom->ma_rr_n_routes = n_routes32;
    atom->ma_rr_target_offset = offset;
    atom->ma_rr_targets = targets;
    atom->ma_rr_probability = prob;
    atom->ma_rr_n_ions = n_ions;
    atom->ma_rr_n_mapped = n_mapped;
    atom->ma_rr_loaded = 1;
    printf("  [MA-RADRECOMB] target data: %d ions, %d levels mapped%s\n",
           n_ions, n_mapped,
           n_scrub ? " (scrubbed out-of-range entries)" : "");
    return 0;
}

/* Load the real Fe III collisional-strength table (Zhang 1996), imported from
 * CMFGEN's FeIII_COL_DATA (19apr23) by scripts/build_feiii_coldata.py.
 * Binary layout (little-endian):
 *   uint32 magic   = 0x46454333 ('FEC3')
 *   uint32 version = 1
 *   int32  Z, ion               (26, 2)
 *   int32  n_trans, n_temp      (n_temp==20)
 *   int32  n_levels_ref         (osc_data level count, sanity vs atom)
 *   double T_grid_K[n_temp]     (CMFGEN 10^4-K grid scaled to K)
 *   record[n_trans]: { int32 i_low, int32 i_high, double omega[n_temp] }
 *     i_low/i_high are Lumina level_number (== CMFGEN osc energy rank),
 *     ordered so i_low is the lower-energy level.
 * Returns 0 on success (atom->feiii_col_loaded=1), -1 on any problem
 * (feiii_col_loaded stays 0 -> assembler falls back to van Regemorter). */
int load_feiii_coldata(AtomicData *atom, const char *path) {
    atom->feiii_col_loaded = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  FeIII col_data: %s not found -> van Regemorter fallback\n", path);
        return -1;
    }
    uint32_t magic = 0, version = 0;
    int32_t Z = 0, ion = 0, n_trans = 0, n_temp = 0, n_lev_ref = 0;
    if (fread(&magic, 4, 1, fp) != 1 || fread(&version, 4, 1, fp) != 1 ||
        fread(&Z, 4, 1, fp) != 1 || fread(&ion, 4, 1, fp) != 1 ||
        fread(&n_trans, 4, 1, fp) != 1 || fread(&n_temp, 4, 1, fp) != 1 ||
        fread(&n_lev_ref, 4, 1, fp) != 1) {
        fprintf(stderr, "ERROR: %s header read failed\n", path);
        fclose(fp); return -1;
    }
    if (magic != 0x46454333u || version != 1u) {
        fprintf(stderr, "ERROR: %s bad magic/version (0x%08x v%u)\n", path, magic, version);
        fclose(fp); return -1;
    }
    if (Z != 26 || ion != 2) {
        fprintf(stderr, "ERROR: %s Z=%d ion=%d != (26,2)\n", path, Z, ion);
        fclose(fp); return -1;
    }
    if (n_trans <= 0 || n_temp <= 0 || n_temp > 256) {
        fprintf(stderr, "ERROR: %s bad n_trans=%d n_temp=%d\n", path, n_trans, n_temp);
        fclose(fp); return -1;
    }
    /* Sanity: the reference level count must match this atom's Fe III level
     * count, else the level_number indices don't correspond (fail-closed). */
    int feiii_nlev = 0;
    for (int l = 0; l < atom->n_levels; l++)
        if (atom->level_Z[l] == 26 && atom->level_ion[l] == 2) feiii_nlev++;
    if (n_lev_ref != feiii_nlev) {
        fprintf(stderr, "ERROR: %s n_levels_ref=%d != atom FeIII levels=%d "
                "(level_number map mismatch)\n", path, n_lev_ref, feiii_nlev);
        fclose(fp); return -1;
    }

    double *tgrid = (double *)malloc((size_t)n_temp * sizeof(double));
    int    *lo    = (int *)malloc((size_t)n_trans * sizeof(int));
    int    *hi    = (int *)malloc((size_t)n_trans * sizeof(int));
    double *omega = (double *)malloc((size_t)n_trans * (size_t)n_temp * sizeof(double));
    if (!tgrid || !lo || !hi || !omega) {
        fprintf(stderr, "ERROR: %s alloc failed\n", path);
        free(tgrid); free(lo); free(hi); free(omega); fclose(fp); return -1;
    }
    if (fread(tgrid, sizeof(double), (size_t)n_temp, fp) != (size_t)n_temp) {
        fprintf(stderr, "ERROR: %s T_grid read failed\n", path);
        free(tgrid); free(lo); free(hi); free(omega); fclose(fp); return -1;
    }
    int bad = 0;
    for (int t = 0; t < n_trans; t++) {
        int32_t il = 0, ih = 0;
        if (fread(&il, 4, 1, fp) != 1 || fread(&ih, 4, 1, fp) != 1 ||
            fread(&omega[(size_t)t * n_temp], sizeof(double), (size_t)n_temp, fp)
                != (size_t)n_temp) {
            fprintf(stderr, "ERROR: %s record %d read failed\n", path, t);
            bad = 1; break;
        }
        if (il < 0 || ih < 0 || il >= feiii_nlev || ih >= feiii_nlev) {
            fprintf(stderr, "ERROR: %s record %d level idx out of range (%d,%d)\n",
                    path, t, il, ih);
            bad = 1; break;
        }
        lo[t] = il; hi[t] = ih;
    }
    fclose(fp);
    if (bad) { free(tgrid); free(lo); free(hi); free(omega); return -1; }

    atom->feiii_col_tgrid   = tgrid;
    atom->feiii_col_lo      = lo;
    atom->feiii_col_hi      = hi;
    atom->feiii_col_omega   = omega;
    atom->feiii_col_Z       = Z;
    atom->feiii_col_ion     = ion;
    atom->feiii_col_n_trans = n_trans;
    atom->feiii_col_n_temp  = n_temp;
    atom->feiii_col_loaded  = 1;
    printf("  FeIII col_data (Zhang 1996): %d transitions x %d T [%.0f..%.0f K], "
           "%d FeIII levels -> real close-coupling Omega ARMED\n",
           n_trans, n_temp, tgrid[0], tgrid[n_temp - 1], feiii_nlev);
    return 0;
}

/* ARTIS-parity (Group A / A3): generic per-ion close-coupling collision-strength
 * loader. Reads one ion's ige_col_<Z>_<ion0>.bin (built by
 * scripts/build_ige_coldata.py, same rate convention as the Fe III Zhang table)
 * and APPENDS it to atom->col_ion_* (Fe II, Co III, Ni III, ...). The level
 * indices are Lumina level_number == CMFGEN osc energy rank, verified identical
 * by the importer's osc<->levels.csv round-trip.
 *
 * Binary layout (little-endian):
 *   uint32 magic=0x49474331 ('IGC1'), uint32 version=1,
 *   int32  Z, ion0(0-based), n_trans, n_temp, n_levels_ref,
 *   double T_grid_K[n_temp],
 *   record[n_trans]: { int32 i_low, int32 i_high, double omega[n_temp] }.
 *
 * Fail-closed: any malformed header/record, a level-count mismatch against this
 * atom's (Z,ion0) level set, or an out-of-range index -> returns -1 and stores
 * nothing (that ion silently falls to ARTIS's g-scaled Axelrod floor). */
int load_ion_coldata(AtomicData *atom, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  ion col_data: %s not found -> Axelrod floor fallback\n", path);
        return -1;
    }
    uint32_t magic = 0, version = 0;
    int32_t Z = 0, ion0 = 0, n_trans = 0, n_temp = 0, n_lev_ref = 0;
    if (fread(&magic, 4, 1, fp) != 1 || fread(&version, 4, 1, fp) != 1 ||
        fread(&Z, 4, 1, fp) != 1 || fread(&ion0, 4, 1, fp) != 1 ||
        fread(&n_trans, 4, 1, fp) != 1 || fread(&n_temp, 4, 1, fp) != 1 ||
        fread(&n_lev_ref, 4, 1, fp) != 1) {
        fprintf(stderr, "ERROR: %s header read failed\n", path);
        fclose(fp); return -1;
    }
    if (magic != 0x49474331u || version != 1u) {
        fprintf(stderr, "ERROR: %s bad magic/version (0x%08x v%u)\n", path, magic, version);
        fclose(fp); return -1;
    }
    if (n_trans <= 0 || n_temp <= 0 || n_temp > 256) {
        fprintf(stderr, "ERROR: %s bad n_trans=%d n_temp=%d\n", path, n_trans, n_temp);
        fclose(fp); return -1;
    }
    if (atom->ncol_ions >= LUMINA_MAX_COL_IONS) {
        fprintf(stderr, "ERROR: %s exceeds LUMINA_MAX_COL_IONS=%d\n", path, LUMINA_MAX_COL_IONS);
        fclose(fp); return -1;
    }
    /* Sanity: reference level count must equal this atom's (Z,ion0) level count
     * (else the level_number == osc-rank identity does not hold). Fail-closed. */
    int nlev_ion = 0;
    for (int l = 0; l < atom->n_levels; l++)
        if (atom->level_Z[l] == Z && atom->level_ion[l] == ion0) nlev_ion++;
    if (n_lev_ref != nlev_ion) {
        fprintf(stderr, "ERROR: %s n_levels_ref=%d != atom Z=%d ion0=%d levels=%d "
                "(level_number map mismatch)\n", path, n_lev_ref, Z, ion0, nlev_ion);
        fclose(fp); return -1;
    }
    double *tgrid = (double *)malloc((size_t)n_temp * sizeof(double));
    int    *lo    = (int *)malloc((size_t)n_trans * sizeof(int));
    int    *hi    = (int *)malloc((size_t)n_trans * sizeof(int));
    double *omega = (double *)malloc((size_t)n_trans * (size_t)n_temp * sizeof(double));
    if (!tgrid || !lo || !hi || !omega) {
        fprintf(stderr, "ERROR: %s alloc failed\n", path);
        free(tgrid); free(lo); free(hi); free(omega); fclose(fp); return -1;
    }
    if (fread(tgrid, sizeof(double), (size_t)n_temp, fp) != (size_t)n_temp) {
        fprintf(stderr, "ERROR: %s T_grid read failed\n", path);
        free(tgrid); free(lo); free(hi); free(omega); fclose(fp); return -1;
    }
    int bad = 0;
    for (int t = 0; t < n_trans; t++) {
        int32_t il = 0, ih = 0;
        if (fread(&il, 4, 1, fp) != 1 || fread(&ih, 4, 1, fp) != 1 ||
            fread(&omega[(size_t)t * n_temp], sizeof(double), (size_t)n_temp, fp)
                != (size_t)n_temp) {
            fprintf(stderr, "ERROR: %s record %d read failed\n", path, t);
            bad = 1; break;
        }
        if (il < 0 || ih < 0 || il >= nlev_ion || ih >= nlev_ion) {
            fprintf(stderr, "ERROR: %s record %d level idx out of range (%d,%d)\n",
                    path, t, il, ih);
            bad = 1; break;
        }
        lo[t] = il; hi[t] = ih;
    }
    fclose(fp);
    if (bad) { free(tgrid); free(lo); free(hi); free(omega); return -1; }

    int c = atom->ncol_ions;
    atom->col_ion_Z[c]       = Z;
    atom->col_ion_stage[c]   = ion0;
    atom->col_ion_n_trans[c] = n_trans;
    atom->col_ion_n_temp[c]  = n_temp;
    atom->col_ion_tgrid[c]   = tgrid;
    atom->col_ion_lo[c]      = lo;
    atom->col_ion_hi[c]      = hi;
    atom->col_ion_omega[c]   = omega;
    atom->ncol_ions          = c + 1;
    printf("  ion col_data Z=%d ion0=%d: %d transitions x %d T [%.0f..%.0f K], "
           "%d levels -> real close-coupling Omega ARMED (slot %d)\n",
           Z, ion0, n_trans, n_temp, tgrid[0], tgrid[n_temp - 1], nlev_ion, c);
    return 0;
}

/* [OMEGA-CMFGEN] Manifest-driven bulk load of EVERY ion for which CMFGEN itself
 * ships a col_data table. Reads <ref_dir>/coldata_cmfgen_manifest.csv and loads
 * the out_bin of every row with status==OK (40 rows / 114952 transitions in
 * data/tardis_reference_toy06_19p48d_sivcaiv). Replaces the 7 hand-listed IGE
 * coolants of the ARTIS-parity banner.
 *
 * Field extraction is index-safe: Z/ion0 are columns 1/2 (before any free-text
 * column) and out_bin/status are the LAST two, so a comma inside a middle note
 * column cannot shift them.
 *
 * Fail-closed: a missing/empty manifest returns -1; any row that load_ion_coldata
 * rejects (bad magic, level-count mismatch, out-of-range level index) makes the
 * whole call return -1 rather than silently running on a partial table set. Rows
 * whose (Z,ion0) is already covered by the Fe III Zhang table are skipped so one
 * level pair never has two sources.
 * Returns the number of ion tables loaded, or -1. */
int load_ion_coldata_manifest(AtomicData *atom, const char *ref_dir) {
    char mpath[1024];
    snprintf(mpath, sizeof(mpath), "%s/coldata_cmfgen_manifest.csv", ref_dir);
    FILE *mf = fopen(mpath, "r");
    if (!mf) {
        fprintf(stderr, "[OMEGA-CMFGEN][ERROR] cannot open %s\n", mpath);
        return -1;
    }
    char lbuf[8192];
    if (!fgets(lbuf, sizeof(lbuf), mf)) {          /* header */
        fprintf(stderr, "[OMEGA-CMFGEN][ERROR] %s: empty manifest\n", mpath);
        fclose(mf);
        return -1;
    }
    int n_ok = 0, n_load = 0, n_fail = 0, n_dup = 0;
    while (fgets(lbuf, sizeof(lbuf), mf)) {
        char *fld[64];
        int nf = 0;
        for (char *p = lbuf; nf < 64; ) {
            fld[nf++] = p;
            char *c = strchr(p, ',');
            if (!c) break;
            *c = '\0';
            p = c + 1;
        }
        if (nf < 4) continue;
        for (char *q = fld[nf - 1]; *q; q++) if (*q == '\n' || *q == '\r') *q = '\0';
        if (strcmp(fld[nf - 1], "OK") != 0) continue;
        n_ok++;
        int Z = atoi(fld[1]), ion0 = atoi(fld[2]);
        const char *bin = fld[nf - 2];
        if (atom->feiii_col_loaded && Z == atom->feiii_col_Z &&
            ion0 == atom->feiii_col_ion) {
            printf("  [OMEGA-CMFGEN] skip %s (Z=%d ion0=%d): already covered by "
                   "feiii_col_zhang.bin\n", bin, Z, ion0);
            n_dup++;
            continue;
        }
        char cp[2048];
        if (strchr(bin, '/')) snprintf(cp, sizeof(cp), "%s", bin);
        else                  snprintf(cp, sizeof(cp), "%s/%s", ref_dir, bin);
        if (load_ion_coldata(atom, cp) == 0) n_load++;
        else {
            n_fail++;
            fprintf(stderr, "[OMEGA-CMFGEN][ERROR] load failed: %s\n", cp);
        }
    }
    fclose(mf);
    printf("  [OMEGA-CMFGEN] manifest %s: %d status==OK rows -> %d loaded, "
           "%d skipped(dup), %d FAILED\n", mpath, n_ok, n_load, n_dup, n_fail);
    if (n_fail > 0 || n_load == 0) return -1;
    return n_load;
}

/* TOP-STAGE CONTINUUM ANCHOR (LUMINA_TOPSTAGE_ANCHOR=1): inject synthetic
 * ground-only IV ion stages so the top NLTE stage III gets a bound-free
 * continuum partner. Without it, top-stage III EXCITED levels have zero
 * continuum coupling and float super-thermal (no recombination drain) ->
 * featureless emergent spectrum. The (III,IV) recombination is the missing
 * thermalizing anchor (validated: thermal levels -> MC features; FORCE_LTE 166340).
 *
 * APPEND at the end of the level arrays: global level indices are UNCHANGED so
 * existing line/macro-transition references stay intact (vs an insert, which
 * would shift them). The synthetic levels are NLTE-scan orphans -- nlte_init
 * finds them by (Z,ion) scan, independent of atom->level_offset (left stale; the
 * bf recomb reads g_ion from level_g directly, so the unused IV partition fn is
 * harmless). Kramers sigma_bf (cmfgen_has_sigma=0). KPACKET must be off (synthetic
 * levels exceed n_macro_levels-sized kpacket arrays; empty macro block otherwise).
 *
 * Increment 1: O IV only (Z=8, ground 2s2 2p 2P, g=6). O is the last NLTE element
 * (slots 28-30 = O I/II/III) so O IV at slot 31 is adjacent (hi=lo+1) -- no
 * explicit-hi pair plumbing needed. S IV/C IV (mid-table III) come next with
 * explicit pair-hi. Gated default-off. */
void inject_topstage_continuum_levels(AtomicData *atom, OpacityState *opacity) {
    const char *e = getenv("LUMINA_TOPSTAGE_ANCHOR");
    if (!(e && atoi(e) != 0)) return;
    /* {Z, ion(0-based, 3=IV), ground g} */
    int syn_Z[]  = {8};
    int syn_ion[]= {3};
    int syn_g[]  = {6};       /* O IV 2s2 2p 2P deg = 6 */
    int n_syn = (int)(sizeof(syn_Z)/sizeof(syn_Z[0]));
    int old_n = atom->n_levels, new_n = old_n + n_syn;

    atom->level_Z          = (int *)   realloc(atom->level_Z,          (size_t)new_n*sizeof(int));
    atom->level_ion        = (int *)   realloc(atom->level_ion,        (size_t)new_n*sizeof(int));
    atom->level_num        = (int *)   realloc(atom->level_num,        (size_t)new_n*sizeof(int));
    atom->level_energy_eV  = (double *)realloc(atom->level_energy_eV,  (size_t)new_n*sizeof(double));
    atom->level_g          = (int *)   realloc(atom->level_g,          (size_t)new_n*sizeof(int));
    atom->level_metastable = (int *)   realloc(atom->level_metastable, (size_t)new_n*sizeof(int));
    atom->level_super      = (int *)   realloc(atom->level_super,      (size_t)new_n*sizeof(int));
    if (atom->level_configuration_sha256) {
        atom->level_configuration_sha256 = (char (*)[65])realloc(
            atom->level_configuration_sha256,
            (size_t)new_n * sizeof(*atom->level_configuration_sha256));
    }
    atom->cmfgen_has_sigma = (int *)   realloc(atom->cmfgen_has_sigma, (size_t)new_n*sizeof(int));
    for (int s = 0; s < n_syn; s++) {
        int l = old_n + s;
        atom->level_Z[l] = syn_Z[s];   atom->level_ion[l] = syn_ion[s];
        atom->level_num[l] = 0;        atom->level_energy_eV[l] = 0.0;
        atom->level_g[l] = syn_g[s];   atom->level_metastable[l] = 1;
        atom->level_super[l] = 0;      atom->cmfgen_has_sigma[l] = 0;
        if (atom->level_configuration_sha256) {
            memset(atom->level_configuration_sha256[l], '0', 64);
            atom->level_configuration_sha256[l][64] = '\0';
        }
    }
    if (atom->cmfgen_sigma_bf && atom->cmfgen_n_freq_bins > 0) {
        size_t nf = (size_t)atom->cmfgen_n_freq_bins;
        atom->cmfgen_sigma_bf = (double *)realloc(atom->cmfgen_sigma_bf,
                                                  (size_t)new_n*nf*sizeof(double));
        memset(&atom->cmfgen_sigma_bf[(size_t)old_n*nf], 0, (size_t)n_syn*nf*sizeof(double));
    }
    atom->n_levels = new_n;

    if (opacity && opacity->macro_block_references) {
        int sentinel = opacity->macro_block_references[opacity->n_macro_levels];
        opacity->macro_block_references = (int *)realloc(opacity->macro_block_references,
                                                         (size_t)(new_n + 1)*sizeof(int));
        for (int s = 0; s <= n_syn; s++)
            opacity->macro_block_references[old_n + s] = sentinel;  /* empty blocks */
        opacity->n_macro_levels = new_n;
    }
    printf("  [TOPSTAGE-ANCHOR] injected %d synthetic IV ground level(s) (O IV g=6); "
           "n_levels %d->%d\n", n_syn, old_n, new_n);
}

void free_atomic_data(AtomicData *atom) {
    free(atom->line_atomic_number);
    free(atom->line_ion_number);
    free(atom->line_level_lower);
    free(atom->line_level_upper);
    free(atom->line_f_lu);
    free(atom->line_wavelength_cm);
    free(atom->line_A_ul);
    free(atom->line_B_lu);
    free(atom->line_B_ul);
    free(atom->line_nu);
    free(atom->level_Z);
    free(atom->level_ion);
    free(atom->level_num);
    free(atom->level_energy_eV);
    free(atom->level_g);
    free(atom->level_metastable);
    free(atom->level_super);
    free(atom->level_configuration_sha256);
    free(atom->level_mult);        /* NULL unless LUMINA_ALPHA_SPINGATE=1 */
    free(atom->ioniz_Z);
    free(atom->ioniz_ion);
    free(atom->ioniz_energy_eV);
    free(atom->zeta_Z);
    free(atom->zeta_ion);
    free(atom->zeta_data);
    free(atom->zeta_temps);
    free(atom->element_Z);
    free(atom->element_mass_amu);
    free(atom->abundances);
    free(atom->elem_ion_offset);
    free(atom->ion_pop_Z);
    free(atom->ion_pop_stage);
    free(atom->level_offset);
    free(atom->ion_number_density);
    free(atom->partition_functions);
    free(atom->cmfgen_has_sigma);
    free(atom->cmfgen_sigma_bf);
    free(atom->ma_rr_target);
    free(atom->ma_rr_target_offset);
    free(atom->ma_rr_targets);
    free(atom->ma_rr_probability);
}

/* ============================================================ */
/* Phase 2 - Step 12: RNG implementation (xoshiro256**)         */
/* ============================================================ */

/* Phase 2 - Step 12: SplitMix64 for seeding */
static uint64_t splitmix64(uint64_t *state) { /* Phase 2 - Step 12 */
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL); /* Phase 2 - Step 12 */
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; /* Phase 2 - Step 12 */
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; /* Phase 2 - Step 12 */
    return z ^ (z >> 31); /* Phase 2 - Step 12 */
}

void rng_init(RNG *rng, uint64_t seed) { /* Phase 2 - Step 12 */
    uint64_t s = seed; /* Phase 2 - Step 12 */
    rng->s[0] = splitmix64(&s); /* Phase 2 - Step 12 */
    rng->s[1] = splitmix64(&s); /* Phase 2 - Step 12 */
    rng->s[2] = splitmix64(&s); /* Phase 2 - Step 12 */
    rng->s[3] = splitmix64(&s); /* Phase 2 - Step 12 */
}

/* Phase 2 - Step 12: Rotate left helper */
static inline uint64_t rotl(const uint64_t x, int k) { /* Phase 2 - Step 12 */
    return (x << k) | (x >> (64 - k)); /* Phase 2 - Step 12 */
}

double rng_uniform(RNG *rng) { /* Phase 2 - Step 12 */
    const uint64_t result = rotl(rng->s[1] * 5, 7) * 9; /* Phase 2 - Step 12 */
    const uint64_t t = rng->s[1] << 17; /* Phase 2 - Step 12 */
    rng->s[2] ^= rng->s[0]; /* Phase 2 - Step 12 */
    rng->s[3] ^= rng->s[1]; /* Phase 2 - Step 12 */
    rng->s[1] ^= rng->s[2]; /* Phase 2 - Step 12 */
    rng->s[0] ^= rng->s[3]; /* Phase 2 - Step 12 */
    rng->s[2] ^= t; /* Phase 2 - Step 12 */
    rng->s[3] = rotl(rng->s[3], 45); /* Phase 2 - Step 12 */
    return (result >> 11) * 0x1.0p-53; /* Phase 2 - Step 12: [0, 1) */
}

double rng_mu(RNG *rng) { /* Phase 2 - Step 12 */
    return 2.0 * rng_uniform(rng) - 1.0; /* Phase 2 - Step 12: [-1, 1) */
}

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */
