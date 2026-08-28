// =============================================================================
//  real_image_blur.cu — applica il filtro a una fotografia vera
//
//  Dimostrazione visiva: carica un JPG in scala di grigi, lo sfoca sulla GPU e
//  salva il risultato in PNG. Riusa kernel, reference CPU e cronometraggio di
//  benchmark.cu invece di ricopiarli.
//
//  A differenza della versione originale il tempo e' misurato con warm-up e
//  ripetizioni, e l'immagine prodotta viene confrontata pixel per pixel con
//  quella calcolata dalla CPU.
//
//  COMPILAZIONE
//      nvcc -O3 -arch=sm_75 real_image_blur.cu -o real_image_blur
//
//  ESECUZIONE
//      ./real_image_blur [input.jpg] [output_blur.png]
// =============================================================================

#define BOXBLUR_NO_MAIN
#include "benchmark.cu"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv)
{
    const char* inPath  = (argc > 1) ? argv[1] : "input.jpg";
    const char* outPath = (argc > 2) ? argv[2] : "output_blur.png";

    const int warmup = 10;
    const int reps   = 50;

    // --- caricamento immagine, forzata a 1 canale (scala di grigi) -----------
    int width = 0, height = 0, channels = 0;
    unsigned char* h_in = stbi_load(inPath, &width, &height, &channels, 1);
    if (!h_in) {
        std::fprintf(stderr, "Impossibile caricare '%s': %s\n", inPath, stbi_failure_reason());
        return EXIT_FAILURE;
    }
    std::printf("Immagine .... %s  %dx%d pixel (%d canali nel file, letta in scala di grigi)\n",
                inPath, width, height, channels);

    const int    pixels = width * height;
    const size_t bytes  = static_cast<size_t>(pixels) * sizeof(unsigned char);

    unsigned char* h_ref = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char* h_gpu = static_cast<unsigned char*>(std::malloc(bytes));
    if (!h_ref || !h_gpu) {
        std::fprintf(stderr, "Allocazione host fallita.\n");
        return EXIT_FAILURE;
    }

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

    // --- correttezza: la GPU deve dare esattamente lo stesso risultato ------
    boxBlurCPU(h_in, h_ref, width, height);

    VerifyResult vNaive  = verifyKernel(launchNaive,  d_out, h_gpu, h_ref, bytes, pixels);
    VerifyResult vShared = verifyKernel(launchShared, d_out, h_gpu, h_ref, bytes, pixels);

    std::printf("Verifica .... naive: %s   shared: %s\n",
                vNaive.ok  ? "OK" : "FALLITA",
                vShared.ok ? "OK" : "FALLITA");
    if (!vNaive.ok)
        std::fprintf(stderr, "  naive:  %d pixel diversi (primo a %d: atteso %d, ottenuto %d)\n",
                     vNaive.mismatches, vNaive.firstIndex, vNaive.expected, vNaive.got);
    if (!vShared.ok)
        std::fprintf(stderr, "  shared: %d pixel diversi (primo a %d: atteso %d, ottenuto %d)\n",
                     vShared.mismatches, vShared.firstIndex, vShared.expected, vShared.got);

    // --- tempi, con riscaldamento e ripetizioni ------------------------------
    Stats cpu    = timeCpu(h_in, h_gpu, width, height, 3);
    Stats naive  = timeGpu(launchNaive,  warmup, reps, start, stop);
    Stats shared = timeGpu(launchShared, warmup, reps, start, stop);

    std::printf("\nTempi (media su %d ripetizioni dopo %d di riscaldamento)\n", reps, warmup);
    std::printf("  CPU sequenziale  %8.3f +- %.3f ms\n", cpu.mean,    cpu.sd);
    std::printf("  GPU naive        %8.4f +- %.4f ms   speed-up %.1fx\n",
                naive.mean, naive.sd, cpu.mean / naive.mean);
    std::printf("  GPU shared       %8.4f +- %.4f ms   speed-up %.1fx\n",
                shared.mean, shared.sd, cpu.mean / shared.mean);
    std::printf("  shared / naive   %8.2fx %s\n", shared.mean / naive.mean,
                (shared.mean > naive.mean) ? "(la shared memory costa di piu')" : "");

    // --- salvataggio del risultato (versione naive, la piu' veloce) ---------
    CHECK(cudaMemset(d_out, 0, bytes));
    launchNaive();
    CHECK_LAUNCH();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    if (!stbi_write_png(outPath, width, height, 1, h_gpu, width)) {
        std::fprintf(stderr, "Scrittura di '%s' fallita.\n", outPath);
        return EXIT_FAILURE;
    }
    std::printf("\nImmagine filtrata salvata in '%s'\n", outPath);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    stbi_image_free(h_in);
    std::free(h_ref);
    std::free(h_gpu);
    CHECK(cudaDeviceReset());

    return (vNaive.ok && vShared.ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
