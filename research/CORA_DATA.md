# Cora dataset for probe_cora.cpp

probe_cora.cpp expects `cora/cora.content` and `cora/cora.cites` (the original
LINQS Cora release) in the working directory. Fetch:

    curl -sSL -o cora.tgz https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
    tar xzf cora.tgz
    cl /O2 /EHsc /std:c++20 probe_cora.cpp && probe_cora.exe

The dataset itself is not committed.
