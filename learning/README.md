# Percorso didattico: dal thread singolo al box blur con shared memory

Ricostruzione guidata, un concetto alla volta, dello stesso risultato gia'
presente in `benchmark.cu` (box blur 3x3: CPU vs kernel naive vs kernel con
shared memory). Ogni file aggiunge un solo concetto nuovo rispetto al
precedente.

| File | Concetto | Stato |
|---|---|---|
| `step1_grid2d.cu` | Griglia 2D di thread, mapping pixel<->thread, boundary check | fatto |
| `step2_naive_blur.cu` | Convoluzione 3x3 letta dalla memoria globale | in corso |
| `step3_shared_blur.cu` | Tiling in shared memory con halo | da fare |
| `step4_bench.cu` | Confronto naive vs shared con metodologia corretta | da fare |

Questi file vanno compilati ed eseguiti su una macchina con GPU (es. Google
Colab, come nel notebook principale): qui non e' disponibile un ambiente
CUDA per compilarli.
