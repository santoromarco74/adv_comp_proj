// =============================================================================
//  step1_grid2d.cu — primo contatto con CUDA: mappare un'immagine su una
//  griglia 2D di thread.
//
//  Questo file NON fa ancora nessun calcolo interessante: il kernel si limita
//  a copiare un pixel da input a output. Serve a fissare, prima di aggiungere
//  qualunque logica di filtro, tre cose:
//
//    1. come un thread scopre "di quale pixel e' responsabile"
//    2. come si dimensiona la griglia di blocchi rispetto all'immagine
//    3. perche' serve un controllo sui bordi (boundary check)
//
//  L'immagine di prova e' volutamente 37x21 (non multipla di 16): cosi' il
//  boundary check entra in gioco fin da subito, invece di sembrare superfluo.
//
//  COMPILAZIONE (su una macchina con GPU, es. Google Colab):
//      nvcc -O2 -arch=sm_75 step1_grid2d.cu -o step1_grid2d
//  ESECUZIONE
//      ./step1_grid2d
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "Errore CUDA in %s:%d -> %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err_));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Lato del blocco di thread. 16x16 = 256 thread per blocco: un valore comune,
// multiplo della dimensione del warp (32), che lascia margine di occupancy
// su quasi tutte le GPU.
#define TILE_SIZE 16

// -----------------------------------------------------------------------------
//  Il kernel
//
//  blockIdx, blockDim, threadIdx sono variabili predefinite che ogni thread
//  vede con un valore diverso:
//    - blockDim  = dimensioni del blocco (qui 16x16), uguale per tutti
//    - blockIdx  = indice del blocco a cui il thread appartiene, nella griglia
//    - threadIdx = indice del thread dentro il proprio blocco
//
//  Combinandoli si ottiene la coordinata globale del pixel, esattamente come
//  si farebbe con due cicli "for" annidati su y e x — ma qui i due cicli sono
//  spariti: ogni iterazione e' un thread fisico che li esegue in parallelo.
// -----------------------------------------------------------------------------
__global__ void identityCopy(const unsigned char* in, unsigned char* out,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // colonna del pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // riga del pixel

    // La griglia e' dimensionata per eccesso (vedi main): se width/height non
    // sono multipli di TILE_SIZE, l'ultimo blocco lancia thread che cadono
    // fuori dall'immagine. Senza questo controllo, quei thread leggerebbero e
    // scriverebbero fuori dall'array allocato: comportamento indefinito, non
    // un errore che nvcc puo' segnalare a compile-time.
    if (x >= width || y >= height) return;

    // L'immagine e' un array 1D in row-major: il pixel (x, y) sta all'indice
    // y * width + x. Non esiste un array 2D "vero" sul device qui: e' una
    // convenzione che host e device devono rispettare allo stesso modo.
    out[y * width + x] = in[y * width + x];
}

int main()
{
    const int width  = 37;   // non multiplo di 16, apposta
    const int height = 21;

    const int    pixels = width * height;
    const size_t bytes  = static_cast<size_t>(pixels) * sizeof(unsigned char);

    // --- immagine sintetica sull'host: ogni pixel ha un valore diverso, cosi'
    //     un errore di indicizzazione si vede subito come pixel sbagliato ---
    unsigned char* h_in  = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char* h_out = static_cast<unsigned char*>(std::malloc(bytes));
    for (int i = 0; i < pixels; i++) h_in[i] = static_cast<unsigned char>(i % 256);

    // --- memoria device + copia host->device -----------------------------
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // --- configurazione della griglia -------------------------------------
    // threads: dimensione fissa del blocco.
    // blocks : quanti blocchi servono per coprire tutta l'immagine, arroton-
    //          dando per eccesso (ceiling division) -> "(n + d - 1) / d".
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((width  + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    int totalThreads = blocks.x * threads.x * blocks.y * threads.y;
    std::printf("Immagine ........ %dx%d = %d pixel\n", width, height, pixels);
    std::printf("Blocco ........... %dx%d = %d thread\n",
                threads.x, threads.y, threads.x * threads.y);
    std::printf("Griglia .......... %dx%d blocchi\n", blocks.x, blocks.y);
    std::printf("Thread lanciati .. %d (%d in piu' del numero di pixel, "
                "escono dal boundary check)\n",
                totalThreads, totalThreads - pixels);

    identityCopy<<<blocks, threads>>>(d_in, d_out, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- verifica: deve essere una copia esatta ---------------------------
    int mismatches = 0;
    for (int i = 0; i < pixels; i++)
        if (h_out[i] != h_in[i]) mismatches++;

    std::printf("Verifica ......... %s (%d differenze su %d pixel)\n",
                mismatches == 0 ? "OK" : "FALLITA", mismatches, pixels);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    std::free(h_in);
    std::free(h_out);

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
