# Affinity Propagation
Affinity Propagation implementace v jazyce **C++20**. Tato implementace
dosahuje warp-rychlostí při paralelním načítání a výpočtu
s využitím následujících technologií:
- **OpenMP**
- **Intel AVX2** SIMD instrukce

## Použití

### Sestavení
```bash
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```

### Spuštění
Program je uzpůsoben pro zpracování CSV souborů s hlavičkou ve formátu:
```
"LabelHeader","FeatureName","FeatureName","FeatureName",...
"Label","feature","feature","feature",...
"Label","feature","feature","feature",...
```

Konfigurace probíhá v `main.cpp` a `consts.h`

```bash
$ cd build
$ ./affinity_propagation
```


## Licence
BSD 2-Clause License

Copyright (c) 2025, Daniel Slavík

[wortelus.eu](https://wortelus.eu)
