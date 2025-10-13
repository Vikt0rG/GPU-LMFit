#pragma once

// Precision
#ifdef MYGPUFIT_DOUBLE
    #define real double
#else
    #define real float
#endif // MYGPUFIT_DOUBLE