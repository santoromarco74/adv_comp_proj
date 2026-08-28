// =============================================================================
//  benchmark.cu — Box blur 3x3: CPU sequenziale vs GPU global memory vs GPU shared memory
//
//  Benchmark con metodologia di misura corretta:
//    - warm-up prima di ogni cronometraggio (elimina il costo di primo lancio)
//    - N ripetizioni con media, deviazione standard, minimo e mediana
//    - verifica di correttezza bit-a-bit contro la reference CPU
//    - controllo esplicito degli errori dopo ogni lancio di kernel
//    - trasferimenti host<->device cronometrati separatamente
//    - speed-up riportato sia "solo kernel" sia "end-to-end"
//
//  COMPILAZIONE (il flag -arch e' obbligatorio: senza, la GPU deve ricompilare
//  il PTX al volo al primo lancio e i tempi risultano falsati)
//
//      nvcc -O3 -arch=sm_75 benchmark.cu -o benchmark      # Tesla T4 (Colab)
//      nvcc -O3 -arch=sm_60 benchmark.cu -o benchmark      # Tesla P100
//      nvcc -O3 -arch=sm_80 benchmark.cu -o benchmark      # A100
//
//  ESECUZIONE
//
//      ./benchmark                                  # default: 512..4096, 100 rip.
//      ./benchmark --sizes 1024,2048 --reps 200
//      ./benchmark --csv risultati.csv              # output per i grafici
//      ./benchmark --order-check                    # prova che l'ordine non conta
//      ./benchmark --help
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
//  Gestione errori
// -----------------------------------------------------------------------------

// Da usare su ogni chiamata alle API CUDA.
#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "Errore CUDA in %s:%d -> %s (codice %d)\n",   \
                         __FILE__, __LINE__, cudaGetErrorString(err_),         \
                         static_cast<int>(err_));                              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Il lancio di un kernel non restituisce un codice di errore: va interrogato
// a parte, subito dopo, altrimenti un lancio fallito passa inosservato.
#define CHECK_LAUNCH() CHECK(cudaGetLastError())

// -----------------------------------------------------------------------------
//  Parametri del tiling
// -----------------------------------------------------------------------------

#define TILE_SIZE  16                    // lato del blocco di thread
#define SHARED_DIM (TILE_SIZE + 2)       // tile + halo di 1 pixel per lato

// -----------------------------------------------------------------------------
//  Cronometro CPU
// -----------------------------------------------------------------------------

static double cpuSeconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return static_cast<double>(tp.tv_sec) + static_cast<double>(tp.tv_usec) * 1.e-6;
}

// -----------------------------------------------------------------------------
//  1. Versione CPU sequenziale (reference)
//
//  Ogni pixel diventa la media dei suoi 9 vicini. Sui bordi la finestra si
//  restringe: si divide per il numero di vicini effettivamente esistenti
//  (count), non per 9. La divisione e' intera, quindi il risultato e' esatto e
//  riproducibile bit-a-bit — ed e' per questo che la GPU puo' essere verificata
//  con un confronto == invece che con una tolleranza.
// -----------------------------------------------------------------------------

static void boxBlurCPU(const unsigned char* in, unsigned char* out,
                       int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0, count = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += in[ny * width + nx];
                        count++;
                    }
                }
            }
            out[y * width + x] = static_cast<unsigned char>(sum / count);
        }
    }
}

// -----------------------------------------------------------------------------
//  2. Kernel GPU "naive": un thread per pixel, letture dalla memoria globale
//
//  Ogni pixel viene riletto fino a 9 volte, una per ciascun vicino. Le riletture
//  pero' vengono quasi sempre servite dalla cache L1 dell'SM, non dalla DRAM.
// -----------------------------------------------------------------------------

__global__ void boxBlurNaive(const unsigned char* in, unsigned char* out,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // L'ultimo blocco sborda quando la larghezza non e' multipla di TILE_SIZE.
    if (x >= width || y >= height) return;

    int sum = 0, count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += in[ny * width + nx];
                count++;
            }
        }
    }
    out[y * width + x] = static_cast<unsigned char>(sum / count);
}

// -----------------------------------------------------------------------------
//  3. Kernel GPU "shared": tiling esplicito in shared memory
//
//  Il blocco copia in shared memory un riquadro 18x18 (il proprio tile 16x16
//  piu' l'halo di un pixel) e poi calcola leggendo solo da li'.
//
//  NOTA: questo kernel e' mantenuto identico a quello del notebook, halo caricato
//  con otto rami condizionali compresi, perche' l'obiettivo del benchmark e'
//  misurare correttamente le due versioni esistenti. Le varianti ottimizzate
//  (caricamento dell'halo senza divergenza, letture uchar4, thread coarsening)
//  vanno in un file separato per non alterare il confronto.
// -----------------------------------------------------------------------------

__global__ void boxBlurShared(const unsigned char* in, unsigned char* out,
                              int width, int height)
{
    __shared__ unsigned char s_data[SHARED_DIM][SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * TILE_SIZE + tx;   // coordinata globale del pixel
    int gy = blockIdx.y * TILE_SIZE + ty;

    int sx = tx + 1;                        // posizione nel tile (offset per l'halo)
    int sy = ty + 1;

    // Le coordinate vengono limitate ai bordi dell'immagine: cosi' i blocchi che
    // sbordano leggono un pixel valido invece di uscire dall'allocazione.
    int cx = min(max(gx, 0), width - 1);
    int cy = min(max(gy, 0), height - 1);

    // (a) pixel centrale: lo carica ogni thread
    s_data[sy][sx] = in[cy * width + cx];

    // (b) halo laterale: solo i thread sul perimetro del blocco
    if (tx == 0)
        s_data[sy][0] = in[cy * width + min(max(gx - 1, 0), width - 1)];
    if (tx == TILE_SIZE - 1)
        s_data[sy][TILE_SIZE + 1] = in[cy * width + min(max(gx + 1, 0), width - 1)];
    if (ty == 0)
        s_data[0][sx] = in[min(max(gy - 1, 0), height - 1) * width + cx];
    if (ty == TILE_SIZE - 1)
        s_data[TILE_SIZE + 1][sx] = in[min(max(gy + 1, 0), height - 1) * width + cx];

    // (c) i quattro angoli dell'halo: un solo thread ciascuno
    if (tx == 0 && ty == 0)
        s_data[0][0] =
            in[min(max(gy - 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)];
    if (tx == TILE_SIZE - 1 && ty == 0)
        s_data[0][TILE_SIZE + 1] =
            in[min(max(gy - 1, 0), height - 1) * width + min(max(gx + 1, 0), width - 1)];
    if (tx == 0 && ty == TILE_SIZE - 1)
        s_data[TILE_SIZE + 1][0] =
            in[min(max(gy + 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)];
    if (tx == TILE_SIZE - 1 && ty == TILE_SIZE - 1)
        s_data[TILE_SIZE + 1][TILE_SIZE + 1] =
            in[min(max(gy + 1, 0), height - 1) * width + min(max(gx + 1, 0), width - 1)];

    // Barriera locale al blocco: un thread interno legge celle di halo scritte da
    // altri thread, quindi deve aspettare che il caricamento sia completo.
    __syncthreads();

    if (gx >= width || gy >= height) return;

    int sum = 0, count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = gx + dx;
            int ny = gy + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += s_data[sy + dy][sx + dx];
                count++;
            }
        }
    }
    out[gy * width + gx] = static_cast<unsigned char>(sum / count);
}

// Kernel vuoto: serve solo a far pagare una volta sola, fuori da ogni finestra
// di cronometraggio, la creazione del contesto e il caricamento del modulo.
__global__ void touchDevice() {}

// -----------------------------------------------------------------------------
//  Statistiche sui campioni
// -----------------------------------------------------------------------------

struct Stats {
    double mean = 0.0;
    double sd   = 0.0;   // deviazione standard campionaria (n-1)
    double min  = 0.0;
    double median = 0.0;
    int    n    = 0;
};

static Stats computeStats(std::vector<float> samples)
{
    Stats s;
    s.n = static_cast<int>(samples.size());
    if (s.n == 0) return s;

    double sum = 0.0;
    for (float v : samples) sum += v;
    s.mean = sum / s.n;

    if (s.n > 1) {
        double acc = 0.0;
        for (float v : samples) {
            double d = v - s.mean;
            acc += d * d;
        }
        s.sd = std::sqrt(acc / (s.n - 1));
    }

    std::sort(samples.begin(), samples.end());
    s.min    = samples.front();
    s.median = (s.n % 2) ? samples[s.n / 2]
                         : 0.5 * (samples[s.n / 2 - 1] + samples[s.n / 2]);
    return s;
}

// Deviazione standard in percentuale sulla media: se supera qualche punto, la
// misura e' rumorosa e va ripetuta con piu' iterazioni.
static double relSd(const Stats& s)
{
    return (s.mean > 0.0) ? 100.0 * s.sd / s.mean : 0.0;
}

// -----------------------------------------------------------------------------
//  Cronometraggio GPU
//
//  Il punto centrale di tutto il file: PRIMA si eseguono 'warmup' lanci a vuoto,
//  POI si misura. Senza questo, il primo lancio del programma si carica addosso
//  il caricamento del modulo e l'eventuale compilazione JIT, e chi viene
//  cronometrato per primo risulta artificialmente lentissimo.
// -----------------------------------------------------------------------------

template <typename LaunchFn>
static Stats timeGpu(LaunchFn launch, int warmup, int reps,
                     cudaEvent_t start, cudaEvent_t stop)
{
    for (int i = 0; i < warmup; i++) {
        launch();
        CHECK_LAUNCH();
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(reps);

    for (int i = 0; i < reps; i++) {
        CHECK(cudaEventRecord(start));
        launch();
        CHECK_LAUNCH();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }
    return computeStats(samples);
}

// La CPU non ha bisogno di warm-up del driver, ma la prima passata scalda le
// cache: anche qui la prima iterazione viene scartata.
static Stats timeCpu(const unsigned char* in, unsigned char* out,
                     int width, int height, int reps)
{
    boxBlurCPU(in, out, width, height);   // passata di riscaldamento, non misurata

    std::vector<float> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; i++) {
        double t0 = cpuSeconds();
        boxBlurCPU(in, out, width, height);
        samples.push_back(static_cast<float>((cpuSeconds() - t0) * 1000.0));
    }
    return computeStats(samples);
}

// -----------------------------------------------------------------------------
//  Verifica di correttezza
//
//  d_out viene azzerato prima del lancio: cosi' un kernel che non scrive nulla
//  (o che scrive solo una parte dell'immagine) fallisce il controllo invece di
//  passare inosservato.
// -----------------------------------------------------------------------------

struct VerifyResult {
    bool ok = false;
    int  mismatches = 0;
    int  firstIndex = -1;
    int  expected = 0;
    int  got = 0;
};

template <typename LaunchFn>
static VerifyResult verifyKernel(LaunchFn launch,
                                 unsigned char* d_out, unsigned char* h_gpu,
                                 const unsigned char* h_ref, size_t bytes,
                                 int pixels)
{
    CHECK(cudaMemset(d_out, 0, bytes));
    launch();
    CHECK_LAUNCH();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    VerifyResult r;
    for (int i = 0; i < pixels; i++) {
        if (h_gpu[i] != h_ref[i]) {
            if (r.firstIndex < 0) {
                r.firstIndex = i;
                r.expected   = h_ref[i];
                r.got        = h_gpu[i];
            }
            r.mismatches++;
        }
    }
    r.ok = (r.mismatches == 0);
    return r;
}

// -----------------------------------------------------------------------------
//  Generazione dell'immagine di prova
//
//  Un pattern deterministico (seme fisso) ma non banale: rumore pseudo-casuale
//  sovrapposto a due gradienti. A differenza di "i % 256" non produce una scala
//  regolare su cui il blur e' quasi l'identita', e a differenza di rand() senza
//  seme e' riproducibile fra un'esecuzione e l'altra.
// -----------------------------------------------------------------------------

static void fillImage(unsigned char* img, int width, int height, unsigned seed)
{
    unsigned state = seed ? seed : 1u;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            state = state * 1664525u + 1013904223u;         // LCG (Numerical Recipes)
            int noise = static_cast<int>((state >> 16) & 0x7F);
            int ramp  = ((x * 255) / width + (y * 255) / height) / 2;
            img[y * width + x] = static_cast<unsigned char>((ramp + noise) & 0xFF);
        }
    }
}

// -----------------------------------------------------------------------------
//  Riga di risultati per una risoluzione
// -----------------------------------------------------------------------------

struct SizeResult {
    int    dim = 0;
    Stats  cpu, h2d, naive, shared, d2h;
    bool   okNaive = false;
    bool   okShared = false;
};

// -----------------------------------------------------------------------------
//  Opzioni da riga di comando
// -----------------------------------------------------------------------------

struct Options {
    std::vector<int> sizes { 512, 1024, 2048, 4096 };
    int  reps       = 100;
    int  warmup     = 10;
    int  cpuReps    = 3;
    unsigned seed   = 42;
    std::string csv;
    bool orderCheck = false;
};

static void printUsage(const char* prog)
{
    std::printf(
        "Uso: %s [opzioni]\n\n"
        "  --sizes L1,L2,...   lati delle immagini quadrate  (default 512,1024,2048,4096)\n"
        "  --reps N            ripetizioni cronometrate per kernel      (default 100)\n"
        "  --warmup N          lanci a vuoto prima di misurare          (default 10)\n"
        "  --cpu-reps N        ripetizioni della versione CPU           (default 3)\n"
        "  --seed N            seme dell'immagine sintetica             (default 42)\n"
        "  --csv FILE          scrive i risultati in CSV (usare - per stdout)\n"
        "  --order-check       rimisura i kernel in ordine invertito e confronta\n"
        "  --help              mostra questo messaggio\n",
        prog);
}

static bool parseSizes(const char* arg, std::vector<int>& out)
{
    out.clear();
    std::string s(arg);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        std::string tok = s.substr(pos, comma - pos);
        if (tok.empty()) return false;
        int v = std::atoi(tok.c_str());
        if (v <= 0) return false;
        out.push_back(v);
        if (comma == s.size()) break;
        pos = comma + 1;
    }
    return !out.empty();
}

static bool parseArgs(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool hasNext = (i + 1 < argc);

        if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (a == "--sizes" && hasNext) {
            if (!parseSizes(argv[++i], o.sizes)) {
                std::fprintf(stderr, "Lista di dimensioni non valida.\n");
                return false;
            }
        } else if (a == "--reps" && hasNext) {
            o.reps = std::atoi(argv[++i]);
        } else if (a == "--warmup" && hasNext) {
            o.warmup = std::atoi(argv[++i]);
        } else if (a == "--cpu-reps" && hasNext) {
            o.cpuReps = std::atoi(argv[++i]);
        } else if (a == "--seed" && hasNext) {
            o.seed = static_cast<unsigned>(std::strtoul(argv[++i], NULL, 10));
        } else if (a == "--csv" && hasNext) {
            o.csv = argv[++i];
        } else if (a == "--order-check") {
            o.orderCheck = true;
        } else {
            std::fprintf(stderr, "Opzione non riconosciuta: %s\n\n", a.c_str());
            printUsage(argv[0]);
            return false;
        }
    }

    if (o.reps < 1 || o.warmup < 0 || o.cpuReps < 1) {
        std::fprintf(stderr, "reps e cpu-reps devono essere >= 1, warmup >= 0.\n");
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
//  Misura di una singola risoluzione
// -----------------------------------------------------------------------------

static SizeResult runSize(int dim, const Options& opt)
{
    SizeResult res;
    res.dim = dim;

    const int    width  = dim;
    const int    height = dim;
    const int    pixels = width * height;
    const size_t bytes  = static_cast<size_t>(pixels) * sizeof(unsigned char);

    // --- memoria host ---------------------------------------------------------
    unsigned char* h_in  = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char* h_ref = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char* h_gpu = static_cast<unsigned char*>(std::malloc(bytes));
    if (!h_in || !h_ref || !h_gpu) {
        std::fprintf(stderr, "Allocazione host fallita per %dx%d.\n", dim, dim);
        std::exit(EXIT_FAILURE);
    }
    fillImage(h_in, width, height, opt.seed);

    // --- memoria device -------------------------------------------------------
    unsigned char *d_in = NULL, *d_out = NULL;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((width  + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    auto launchNaive  = [&] { boxBlurNaive <<<blocks, threads>>>(d_in, d_out, width, height); };
    auto launchShared = [&] { boxBlurShared<<<blocks, threads>>>(d_in, d_out, width, height); };

    // --- 1. reference CPU + verifica -----------------------------------------
    boxBlurCPU(h_in, h_ref, width, height);

    VerifyResult vNaive  = verifyKernel(launchNaive,  d_out, h_gpu, h_ref, bytes, pixels);
    VerifyResult vShared = verifyKernel(launchShared, d_out, h_gpu, h_ref, bytes, pixels);
    res.okNaive  = vNaive.ok;
    res.okShared = vShared.ok;

    if (!vNaive.ok)
        std::fprintf(stderr,
                     "  [%dx%d] naive:  %d pixel diversi, primo a %d (atteso %d, ottenuto %d)\n",
                     dim, dim, vNaive.mismatches, vNaive.firstIndex, vNaive.expected, vNaive.got);
    if (!vShared.ok)
        std::fprintf(stderr,
                     "  [%dx%d] shared: %d pixel diversi, primo a %d (atteso %d, ottenuto %d)\n",
                     dim, dim, vShared.mismatches, vShared.firstIndex, vShared.expected, vShared.got);

    // --- 2. tempi ------------------------------------------------------------
    res.cpu    = timeCpu(h_in, h_gpu, width, height, opt.cpuReps);
    res.h2d    = timeGpu([&] { CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice)); },
                         opt.warmup, opt.reps, start, stop);
    res.naive  = timeGpu(launchNaive,  opt.warmup, opt.reps, start, stop);
    res.shared = timeGpu(launchShared, opt.warmup, opt.reps, start, stop);
    res.d2h    = timeGpu([&] { CHECK(cudaMemcpyAsync(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost)); },
                         opt.warmup, opt.reps, start, stop);

    // --- 3. controllo dell'ordine di misura ----------------------------------
    // Con il warm-up l'ordine non deve piu' contare. Rimisurando a kernel
    // invertiti si ottiene la prova da mettere nel report.
    if (opt.orderCheck) {
        Stats sharedFirst = timeGpu(launchShared, opt.warmup, opt.reps, start, stop);
        Stats naiveSecond = timeGpu(launchNaive,  opt.warmup, opt.reps, start, stop);

        double dN = 100.0 * (naiveSecond.mean - res.naive.mean)  / res.naive.mean;
        double dS = 100.0 * (sharedFirst.mean - res.shared.mean) / res.shared.mean;

        std::printf("  controllo ordine  naive %.4f -> %.4f ms (%+.1f%%)   "
                    "shared %.4f -> %.4f ms (%+.1f%%)\n",
                    res.naive.mean, naiveSecond.mean, dN,
                    res.shared.mean, sharedFirst.mean, dS);
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    std::free(h_in);
    std::free(h_ref);
    std::free(h_gpu);

    return res;
}

// -----------------------------------------------------------------------------
//  Stampa
// -----------------------------------------------------------------------------

static void printDeviceInfo()
{
    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));

    // Banda di picco teorica: bus in bit / 8, clock in kHz, DDR quindi x2.
    double peakGBs = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1.0e6;

    std::printf("GPU ............ %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    std::printf("Multiprocessori  %d\n", p.multiProcessorCount);
    std::printf("Shared memory .. %zu KB per blocco, %zu KB per multiprocessore\n",
                p.sharedMemPerBlock / 1024, p.sharedMemPerMultiprocessor / 1024);
    std::printf("Banda di picco .. %.1f GB/s\n", peakGBs);
}

static void printTable(const std::vector<SizeResult>& all)
{
    std::printf("\n");
    std::printf("Tempi in millisecondi: media +/- deviazione standard\n");
    std::printf("%-11s %18s %18s %18s %18s %18s\n",
                "Risoluzione", "CPU", "copia H->D", "GPU naive", "GPU shared", "copia D->H");
    std::printf("%s\n", std::string(107, '-').c_str());

    for (const SizeResult& r : all) {
        char dims[24];
        std::snprintf(dims, sizeof(dims), "%dx%d", r.dim, r.dim);
        std::printf("%-11s %11.3f+-%-6.3f %11.4f+-%-6.4f %11.4f+-%-6.4f %11.4f+-%-6.4f %11.4f+-%-6.4f\n",
                    dims,
                    r.cpu.mean,    r.cpu.sd,
                    r.h2d.mean,    r.h2d.sd,
                    r.naive.mean,  r.naive.sd,
                    r.shared.mean, r.shared.sd,
                    r.d2h.mean,    r.d2h.sd);
    }

    // --- speed-up ------------------------------------------------------------
    std::printf("\n");
    std::printf("Speed-up rispetto alla CPU sequenziale\n");
    std::printf("%-11s %13s %13s %13s %13s %15s\n",
                "Risoluzione", "naive kern.", "shared kern.", "naive e2e", "shared e2e", "shared/naive");
    std::printf("%s\n", std::string(84, '-').c_str());

    for (const SizeResult& r : all) {
        char dims[24];
        std::snprintf(dims, sizeof(dims), "%dx%d", r.dim, r.dim);

        double e2eNaive  = r.h2d.mean + r.naive.mean  + r.d2h.mean;
        double e2eShared = r.h2d.mean + r.shared.mean + r.d2h.mean;

        std::printf("%-11s %12.1fx %12.1fx %12.1fx %12.1fx %14.2fx\n",
                    dims,
                    r.cpu.mean / r.naive.mean,
                    r.cpu.mean / r.shared.mean,
                    r.cpu.mean / e2eNaive,
                    r.cpu.mean / e2eShared,
                    r.naive.mean / r.shared.mean);
    }
    std::printf("\n  kern. = solo tempo di calcolo sulla GPU\n");
    std::printf("  e2e   = end-to-end, copia andata + kernel + copia ritorno\n");
    std::printf("  shared/naive > 1 significa che la shared memory conviene\n");

    // --- banda effettiva -----------------------------------------------------
    std::printf("\n");
    std::printf("Banda di memoria effettiva dei kernel (letture + scritture / tempo)\n");
    std::printf("%-11s %14s %14s %12s\n", "Risoluzione", "naive", "shared", "rumore");
    std::printf("%s\n", std::string(54, '-').c_str());

    for (const SizeResult& r : all) {
        char dims[24];
        std::snprintf(dims, sizeof(dims), "%dx%d", r.dim, r.dim);

        double movedGB = 2.0 * static_cast<double>(r.dim) * r.dim / 1.0e9;  // in + out
        double bwNaive  = movedGB / (r.naive.mean  / 1000.0);
        double bwShared = movedGB / (r.shared.mean / 1000.0);
        double noise    = std::max(relSd(r.naive), relSd(r.shared));

        std::printf("%-11s %10.1f GB/s %10.1f GB/s %10.1f %%\n",
                    dims, bwNaive, bwShared, noise);
    }
    std::printf("\n  rumore = deviazione standard piu' alta fra i due kernel, in %% della media\n");
    std::printf("           sopra il 5%% conviene aumentare --reps\n");

    // --- verifica ------------------------------------------------------------
    std::printf("\n");
    std::printf("Verifica di correttezza (confronto bit-a-bit con la reference CPU)\n");
    for (const SizeResult& r : all) {
        std::printf("  %dx%d   naive: %-8s shared: %-8s\n",
                    r.dim, r.dim,
                    r.okNaive  ? "OK"   : "FALLITA",
                    r.okShared ? "OK"   : "FALLITA");
    }
}

static void writeCsv(const std::vector<SizeResult>& all, const std::string& path)
{
    FILE* f = (path == "-") ? stdout : std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "Impossibile aprire %s in scrittura.\n", path.c_str());
        return;
    }

    std::fprintf(f,
        "dim,cpu_ms,cpu_sd,h2d_ms,h2d_sd,naive_ms,naive_sd,shared_ms,shared_sd,"
        "d2h_ms,d2h_sd,speedup_naive_kernel,speedup_shared_kernel,"
        "speedup_naive_e2e,speedup_shared_e2e,shared_vs_naive,verify_naive,verify_shared\n");

    for (const SizeResult& r : all) {
        double e2eNaive  = r.h2d.mean + r.naive.mean  + r.d2h.mean;
        double e2eShared = r.h2d.mean + r.shared.mean + r.d2h.mean;

        std::fprintf(f,
            "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
            r.dim,
            r.cpu.mean,    r.cpu.sd,
            r.h2d.mean,    r.h2d.sd,
            r.naive.mean,  r.naive.sd,
            r.shared.mean, r.shared.sd,
            r.d2h.mean,    r.d2h.sd,
            r.cpu.mean / r.naive.mean,
            r.cpu.mean / r.shared.mean,
            r.cpu.mean / e2eNaive,
            r.cpu.mean / e2eShared,
            r.naive.mean / r.shared.mean,
            r.okNaive ? 1 : 0,
            r.okShared ? 1 : 0);
    }

    if (f != stdout) {
        std::fclose(f);
        std::printf("\nCSV scritto in %s\n", path.c_str());
    }
}

// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    Options opt;
    if (!parseArgs(argc, argv, opt)) return EXIT_FAILURE;

    printDeviceInfo();
    std::printf("Configurazione .. blocco %dx%d, tile in shared memory %dx%d\n",
                TILE_SIZE, TILE_SIZE, SHARED_DIM, SHARED_DIM);
    std::printf("Misura ......... %d ripetizioni dopo %d lanci di riscaldamento, "
                "CPU %d ripetizioni, seme %u\n",
                opt.reps, opt.warmup, opt.cpuReps, opt.seed);

    // Primo lancio del programma: paga qui, fuori da ogni cronometraggio, la
    // creazione del contesto e il caricamento del modulo.
    touchDevice<<<1, 1>>>();
    CHECK_LAUNCH();
    CHECK(cudaDeviceSynchronize());

    std::vector<SizeResult> all;
    for (int dim : opt.sizes) {
        std::printf("\n[%dx%d] in corso...\n", dim, dim);
        all.push_back(runSize(dim, opt));
    }

    printTable(all);

    if (!opt.csv.empty()) writeCsv(all, opt.csv);

    bool allOk = true;
    for (const SizeResult& r : all) allOk = allOk && r.okNaive && r.okShared;

    CHECK(cudaDeviceReset());
    return allOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
