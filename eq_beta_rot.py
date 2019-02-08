"""Equatorial beta plane test
No rotation
Add rotation later


"""
import os
import sys
import time

import numpy as np
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

import logging
logger = logging.getLogger(__name__)


# Grid parameters
Lx = 40
Ly = 10

nx = 400
ny = 100

# Model parameters
A = 0.12
B = 0.394

x_basis = de.Fourier('x',nx,  interval=[-Lx/2, Lx/2])#, dealias=3/2) # Dealias needed or not?
y_basis = de.Chebyshev('y',ny, interval=[-Ly/2, Ly/2])#, dealias=3/2)

domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['u','v','h','s','uy','vy','sy'])
Grav = 9.8
Nu   = 0.1
D    = 0.1
tau  = 5

problem.parameters['g']  = Grav
problem.parameters['nu'] = Nu
problem.parameters['D']  = D
problem.parameters['tau'] = tau

problem.parameters['Lx'] = Lx # Original
problem.parameters['Ly'] = Ly
problem.parameters['β'] = 1
problem.substitutions['f'] = "β*y"
problem.substitutions['vol_avg(A)']   = 'integ(A)/(Lx*Ly)'

problem.add_equation("dt(u) + g*dx(h) - nu*(dx(dx(u)) + dy(uy)) - f*v = -u*dx(u) - v*uy")
problem.add_equation("dt(v) + g*dy(h) - nu*(dx(dx(v)) + dy(vy)) + f*u = -u*dx(v) - v*vy")
problem.add_equation("dt(h) = -h*dx(u) - h*vy - u*dx(h) - v*dy(h)")
problem.add_equation("dt(s) - D*(dx(dx(s)) + dy(sy)) = - s*dx(u) - s*vy - u*dx(s) - v*sy") # passive scalar

problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("sy - dy(s) = 0")

problem.add_bc("left(uy) = 0")  # Actually the bottom.  Free slip.
problem.add_bc("left(v) = 0")   # No through flow
problem.add_bc("left(sy) = 0")   # Tracer BC?
#problem.add_bc("left(hy) = 0")  # Neumann in height

problem.add_bc("right(uy) = 0") # Actually the top. å Free slip.
problem.add_bc("right(v) = 0")#, condition="(nx != 0)")  # No through flow
problem.add_bc("right(sy) = 0")  # Tracer BC?
#problem.add_bc("right(hy) = 0") # Neumann in height

# Build solver
ts = de.timesteppers.RK443
solver =  problem.build_solver(ts)
logger.info('Solver built')

# Initial Conditions
eta = domain.new_field()

u  = solver.state['u']
uy = solver.state['uy']
v  = solver.state['v']
vy = solver.state['vy']
h  = solver.state['h']
s  = solver.state['s']
sy = solver.state['sy']

xx, yy = domain.grids()
xone = np.ones(xx.shape)
yone = np.ones(yy.shape)
xyone = xone*yone
eta['g'] = A/np.cosh(B*xx)**2
deta = eta.differentiate('x')
h['g'] = 0.5*xyone + eta['g'] * (3 + 6*yy**2)/4. * np.exp(-yy**2/2) / Grav
u['g'] = 0*xyone #eta['g'] * (-9 + 6*yy**2)/4. * np.exp(-yy**2/2)
v['g'] = 0*xyone #2 * deta['g'] * yy * np.exp(-yy**2/2)
s['g'] = xyone#2 * deta['g'] * yy * np.exp(-yy**2/2)
#v['g'] = np.zeros(y.size)#amp*np.sin(2.0*np.pi*x/Lx)*np.exp(-(y*y)/(sigma*sigma))
#s['g'] = np.ones(y.size)
u.differentiate('y',out=uy)
v.differentiate('y',out=vy)
s.differentiate('y',out=sy)

# Time Stepping
#solver.stop_sim_time = 100
#solver.stop_wall_time = np.inf
#solver.stop_iteration = 10000000
#dt = 0.025
#logger.info('Solver built')

solver.stop_sim_time = 10.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf #h500 #np.inf
#dt = 0.025
initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.5)
cfl.add_velocities(('u','v'))
#logger.info('Solver built')
dt = cfl.compute_dt()

data_dir = 'eq_beta'
if domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Analysis
snapshots = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1., max_writes=np.inf)
snapshots.add_system(solver.state)

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'),iter=10, max_writes=np.inf)
timeseries.add_task("vol_avg(sqrt(u*u))",name='urms')
timeseries.add_task("vol_avg(sqrt(v*v))",name='vrms')

analysis_tasks = [snapshots, timeseries]

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        #dt = cfl.compute_dt()
        dt = 0.025
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

    for task in analysis_tasks:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)

