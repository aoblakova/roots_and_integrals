# roots_and_integrals
The source code for the numerical results of Oblakova et al. "Roots, symmetry and contour integrals in queueing systems" submitted to SIAM Journal on Applied Mathematics.

## Description
Compares different approaches for solving the linear system of equation for x<sub>j</sub>:

<img src="https://render.githubusercontent.com/render/math?math=\sum_{j=0}^n x_j\, f_j(1) = D'(1),">,

<img src="https://render.githubusercontent.com/render/math?math=\sum_{j=0}^n x_j\, f_j(z_k) = 0,\ k=1,\dots,n">,

where f<sub>j</sub> and D(z) are known functions, and z<sub>k</sub> are the zeroes of D(z) inside the unit disk. Such systems of equations occur in queueing models. 
For a special case of the bulk-service queue with binomial arrivals, the project plots the error compared to the explicit solution.


## License
Copyright 2020 Anna Oblakova

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
