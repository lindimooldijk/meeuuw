# meeuuw
Mantle modelling Early Earth Utrecht University Work-in-progress

Code description:
- FEM
- Q2Q1 finite element pair for velocity
- Q2 finite element for temperature
- 2d 
- linear viscous
- particle-in-cell (passive)
- Runge-Kutta in space: 1st, 2nd, 4th order
- export to vtu file
- direct solver for both linear systems
- Crank-Nicolson time scheme for T equation

to do:
- poisson disc distribution for particles
- more accurate heat flux calculations
- SUPG and/or Lenardic & Kaula filter

