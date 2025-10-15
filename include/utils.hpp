#pragma once

#include "types.hpp"
#include <cstddef>

// Define a function type
typedef float (*ModelFuncType)(const real* x, const real* parameterArray);