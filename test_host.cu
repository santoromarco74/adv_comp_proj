// Test host-only: emula i due kernel replicando esattamente la loro logica di
// indicizzazione, per verificare senza GPU che l'halo copra tutto il tile e che
// i risultati coincidano con la reference CPU anche su dimensioni non multiple
// di TILE_SIZE.
#define main benchmark_main_unused
#include "benchmark.cu"
#undef main

#include <cassert>

static int failures = 0;
static void expect(bool cond, const char* what) {
    if (!cond) { std::printf("  FAIL: %s\n", what); failures++; }
    else       { std::printf("  ok  : %s\n", what); }
}

// --- emulazione del kernel naive --------------------------------------------
static void emulateNaive(const unsigned char* in, unsigned char* out, int W, int H)
{
    dim3 threads(TILE_SIZE, TILE_SIZE);
    int gridX = (W + TILE_SIZE - 1) / TILE_SIZE;
    int gridY = (H + TILE_SIZE - 1) / TILE_SIZE;

    for (int by = 0; by < gridY; by++)
    for (int bx = 0; bx < gridX; bx++)
    for (int ty = 0; ty < (int)threads.y; ty++)
    for (int tx = 0; tx < (int)threads.x; tx++) {
        int x = bx * TILE_SIZE + tx;
        int y = by * TILE_SIZE + ty;
        if (x >= W || y >= H) continue;
        int sum = 0, count = 0;
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) { sum += in[ny*W+nx]; count++; }
        }
        out[y*W+x] = (unsigned char)(sum / count);
    }
}

// --- emulazione del kernel shared -------------------------------------------
// Traccia anche quali celle di shared memory vengono scritte, per scoprire
// eventuali letture di celle mai inizializzate.
static void emulateShared(const unsigned char* in, unsigned char* out, int W, int H,
                          int* uninitReads)
{
    *uninitReads = 0;
    int gridX = (W + TILE_SIZE - 1) / TILE_SIZE;
    int gridY = (H + TILE_SIZE - 1) / TILE_SIZE;

    for (int by = 0; by < gridY; by++)
    for (int bx = 0; bx < gridX; bx++) {
        unsigned char s[SHARED_DIM][SHARED_DIM];
        bool written[SHARED_DIM][SHARED_DIM];
        std::memset(s, 0, sizeof(s));
        std::memset(written, 0, sizeof(written));

        // fase 1: caricamento (tutti i thread), poi __syncthreads()
        for (int ty = 0; ty < TILE_SIZE; ty++)
        for (int tx = 0; tx < TILE_SIZE; tx++) {
            int gx = bx * TILE_SIZE + tx;
            int gy = by * TILE_SIZE + ty;
            int sx = tx + 1, sy = ty + 1;
            int cx = std::min(std::max(gx, 0), W - 1);
            int cy = std::min(std::max(gy, 0), H - 1);

            s[sy][sx] = in[cy*W+cx];                          written[sy][sx] = true;

            if (tx == 0) { s[sy][0] = in[cy*W + std::min(std::max(gx-1,0),W-1)]; written[sy][0] = true; }
            if (tx == TILE_SIZE-1) { s[sy][TILE_SIZE+1] = in[cy*W + std::min(std::max(gx+1,0),W-1)]; written[sy][TILE_SIZE+1] = true; }
            if (ty == 0) { s[0][sx] = in[std::min(std::max(gy-1,0),H-1)*W + cx]; written[0][sx] = true; }
            if (ty == TILE_SIZE-1) { s[TILE_SIZE+1][sx] = in[std::min(std::max(gy+1,0),H-1)*W + cx]; written[TILE_SIZE+1][sx] = true; }

            if (tx == 0 && ty == 0) { s[0][0] = in[std::min(std::max(gy-1,0),H-1)*W + std::min(std::max(gx-1,0),W-1)]; written[0][0] = true; }
            if (tx == TILE_SIZE-1 && ty == 0) { s[0][TILE_SIZE+1] = in[std::min(std::max(gy-1,0),H-1)*W + std::min(std::max(gx+1,0),W-1)]; written[0][TILE_SIZE+1] = true; }
            if (tx == 0 && ty == TILE_SIZE-1) { s[TILE_SIZE+1][0] = in[std::min(std::max(gy+1,0),H-1)*W + std::min(std::max(gx-1,0),W-1)]; written[TILE_SIZE+1][0] = true; }
            if (tx == TILE_SIZE-1 && ty == TILE_SIZE-1) { s[TILE_SIZE+1][TILE_SIZE+1] = in[std::min(std::max(gy+1,0),H-1)*W + std::min(std::max(gx+1,0),W-1)]; written[TILE_SIZE+1][TILE_SIZE+1] = true; }
        }

        // fase 2: calcolo
        for (int ty = 0; ty < TILE_SIZE; ty++)
        for (int tx = 0; tx < TILE_SIZE; tx++) {
            int gx = bx * TILE_SIZE + tx;
            int gy = by * TILE_SIZE + ty;
            int sx = tx + 1, sy = ty + 1;
            if (gx >= W || gy >= H) continue;

            int sum = 0, count = 0;
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int nx = gx + dx, ny = gy + dy;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    if (!written[sy+dy][sx+dx]) (*uninitReads)++;
                    sum += s[sy+dy][sx+dx];
                    count++;
                }
            }
            out[gy*W+gx] = (unsigned char)(sum / count);
        }
    }
}

static void testSize(int W, int H)
{
    std::printf("\n--- immagine %dx%d ---\n", W, H);
    int px = W * H;
    std::vector<unsigned char> in(px), ref(px), na(px, 0xAB), sh(px, 0xAB);
    fillImage(in.data(), W, H, 42);

    boxBlurCPU(in.data(), ref.data(), W, H);
    emulateNaive(in.data(), na.data(), W, H);
    int uninit = 0;
    emulateShared(in.data(), sh.data(), W, H, &uninit);

    int dn = 0, ds = 0, firstS = -1;
    for (int i = 0; i < px; i++) {
        if (na[i] != ref[i]) dn++;
        if (sh[i] != ref[i]) { if (firstS < 0) firstS = i; ds++; }
    }
    char msg[160];
    std::snprintf(msg, sizeof(msg), "naive  == reference CPU  (%d differenze)", dn);
    expect(dn == 0, msg);
    std::snprintf(msg, sizeof(msg), "shared == reference CPU  (%d differenze%s)", ds,
                  firstS >= 0 ? ", prima a idx sopra" : "");
    expect(ds == 0, msg);
    if (firstS >= 0)
        std::printf("        primo pixel diverso: idx=%d (x=%d,y=%d) atteso=%d ottenuto=%d\n",
                    firstS, firstS % W, firstS / W, ref[firstS], sh[firstS]);
    std::snprintf(msg, sizeof(msg), "nessuna lettura di shared memory non inizializzata (%d)", uninit);
    expect(uninit == 0, msg);
}

int main()
{
    std::printf("=== fillImage: determinismo e distribuzione ===\n");
    {
        std::vector<unsigned char> a(4096), b(4096);
        fillImage(a.data(), 64, 64, 42);
        fillImage(b.data(), 64, 64, 42);
        expect(std::memcmp(a.data(), b.data(), 4096) == 0, "stesso seme -> stessa immagine");
        fillImage(b.data(), 64, 64, 43);
        expect(std::memcmp(a.data(), b.data(), 4096) != 0, "semi diversi -> immagini diverse");
        int distinct[256] = {0}, used = 0;
        for (unsigned char v : a) distinct[v]++;
        for (int i = 0; i < 256; i++) if (distinct[i]) used++;
        char m[96]; std::snprintf(m, sizeof(m), "usa molti livelli di grigio (%d/256)", used);
        expect(used > 100, m);
    }

    std::printf("\n=== computeStats ===\n");
    {
        Stats s = computeStats({2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f});
        expect(std::fabs(s.mean - 5.0) < 1e-6, "media = 5");
        expect(std::fabs(s.sd - 2.13809) < 1e-4, "deviazione standard campionaria = 2.1381");
        expect(std::fabs(s.median - 4.5) < 1e-6, "mediana = 4.5");
        expect(std::fabs(s.min - 2.0) < 1e-6, "minimo = 2");
        Stats one = computeStats({3.0f});
        expect(one.sd == 0.0 && one.mean == 3.0, "campione singolo non divide per zero");
    }

    std::printf("\n=== parseSizes ===\n");
    {
        std::vector<int> v;
        expect(parseSizes("512,1024,2048", v) && v.size() == 3 && v[2] == 2048, "lista valida");
        expect(parseSizes("777", v) && v.size() == 1 && v[0] == 777, "valore singolo");
        expect(!parseSizes("512,,1024", v), "rifiuta campo vuoto");
        expect(!parseSizes("512,-8", v), "rifiuta valore negativo");
        expect(!parseSizes("", v), "rifiuta stringa vuota");
    }

    std::printf("\n=== equivalenza dei kernel (emulati sull'host) ===\n");
    testSize(64, 64);        // multiplo esatto di 16
    testSize(128, 96);       // rettangolare, multiplo
    testSize(100, 37);       // NON multiplo: l'ultimo blocco sborda
    testSize(17, 17);        // poco piu' di un blocco
    testSize(16, 16);        // esattamente un blocco
    testSize(1, 1);          // caso degenere
    testSize(3, 200);        // molto stretta

    std::printf("\n============================\n");
    if (failures == 0) std::printf("TUTTI I TEST SUPERATI\n");
    else               std::printf("%d TEST FALLITI\n", failures);
    return failures == 0 ? 0 : 1;
}
