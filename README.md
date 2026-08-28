# Box blur 3×3 su GPU — CPU vs memoria globale vs shared memory

Progetto per il corso **Advanced Computer Architecture**, traccia *Image and video
processing: filters on images*.

Un filtro box blur 3×3 implementato in tre versioni — CPU sequenziale, GPU su
memoria globale, GPU con tiling in shared memory — verificate bit per bit e
confrontate su Tesla T4.

📄 **[Leggi il report completo →](REPORT.md)**

---

## Risultato in breve

| | |
|---|---|
| Speed-up del solo calcolo, GPU vs CPU sequenziale | **~240×** |
| Speed-up end-to-end, con i trasferimenti | **~20×** |
| Quota del tempo end-to-end spesa sul bus PCIe | **89–93 %** |
| Banda di memoria effettivamente sfruttata | **9–17 %** dei 320 GB/s |
| Tiling in shared memory rispetto alla versione naive | **1,35–1,53× più lento** |

L'ultima riga è il risultato più interessante del lavoro. In uno stencil 3×3 il
riuso dei dati è solo 9× e l'area di lavoro di un blocco è di 324 byte: la cache
L1 esegue già il tiling per conto proprio, quindi la versione esplicita paga
barriera di sincronizzazione, rami divergenti e carico sbilanciato senza
incassare alcun beneficio. La [sezione 7 del report](REPORT.md#7-discussione-perché-il-tiling-non-paga)
argomenta il perché.

![Rapporto fra i tempi dei due kernel](figure/fig2_shared_vs_naive.png)

## File

| File | Contenuto |
|---|---|
| [`REPORT.md`](REPORT.md) | il report: problema, strategie, metodologia, risultati, discussione |
| `benchmark.cu` | i due kernel, il riferimento CPU, misura e verifica |
| `test_host.cu` | verifica di correttezza eseguibile anche senza GPU |
| `real_image_blur.cu` | il filtro applicato a una fotografia reale |
| `adv_comp_proj.ipynb` | notebook Colab che orchestra il tutto |
| `figure/` | grafici e dati grezzi (`risultati.csv`) |

## Come eseguire

Su Google Colab con runtime GPU, apri il notebook ed esegui le celle in ordine.
In locale, con il CUDA toolkit installato:

```bash
nvcc -O3 -arch=sm_75 benchmark.cu -o benchmark     # sm_75 = Tesla T4
./benchmark --csv risultati.csv --order-check --cpu-reps 10

nvcc -O2 -arch=sm_75 test_host.cu -o test_host && ./test_host
```

Il flag `-arch` non è opzionale: senza, la GPU deve compilare il PTX al volo al
primo lancio, e il costo ricade sulla prima misura falsandola.

Tutte le opzioni: `./benchmark --help`.

## Nota sulla metodologia

Ogni tempo è misurato **dopo un warm-up** e mediato su 100 ripetizioni, con
deviazione standard riportata. Non è un dettaglio pedante: senza warm-up il primo
kernel lanciato assorbe il costo una tantum di caricamento del modulo, e le stesse
identiche implementazioni sulla stessa scheda producono misure che differiscono
fino a un fattore 300 — portando alla conclusione opposta. La
[sezione 5 del report](REPORT.md#5-metodologia-di-misura) documenta il problema e
il protocollo adottato.
