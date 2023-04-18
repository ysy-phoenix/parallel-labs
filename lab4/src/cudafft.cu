#include <cufftw.h>
#include <iostream>

using namespace std;

using cutype = cufftDoubleComplex;

ostream &operator<<(ostream &os, const cutype &rhs) {
    os << "(" << rhs.x << ", " << rhs.y << ")";
    return os;
}

