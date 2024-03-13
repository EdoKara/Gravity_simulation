import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numba 
from numba import cuda
import time



# test = np.indices((10,10))
# inds = np.ndindex((10,10,10))
# print(inds)

# a = np.ndarray((10,10,10))
# b = np.indices((2,2))


# xv, yv, zv = np.meshgrid(np.linspace(0,3,3), np.linspace(0,3,3), np.linspace(0,3,3))

def make_model_field(nrows, ncols, nlayers):
    return np.meshgrid(np.linspace(1,nrows,nrows), np.linspace(1,ncols,ncols), np.linspace(1,nlayers,nlayers), indexing='xy')

# np.fromfunction(lambda i,j,k: np.array([i,j,k]) ,(300,300,300))

# test = make_model_field(1000,1000,1000)

# xx, yy, zz = test


class PointMass:
    def __init__(self, x:float, y:float, z:float, mass:float, velocity:tuple[float, float, float] | None) -> None:
        self.x = x;
        self.y = y;
        self.z = z;
        self.mass = mass;
        self.velocity = velocity;
    
# pm=PointMass(4.1,8.1,3.2,1000000)

# xx_t = xx - pm.x
# xx_t_sign = -1*(xx_t/np.abs(xx_t))


# fgx = xx_t_sign*((G*pm.mass)*(1/np.square(xx_t)))

# base_transform = np.diag([1,1,1])
# to_point_coords = base_transform - np.diag([pm.x, pm.y, pm.z])
# to_transform_sign = to_point_coords/np.abs(to_point_coords)

# fgs = np.nan_to_num((G*pm.mass)*(1/np.square(to_point_coords))*to_transform_sign, nan=0.0)

class GravityField:
    def __init__(self, field, pointmasses:list) -> None:
        self.field = field;
        self.pointmasses = pointmasses;
        
def evaluate_gravity_force(gf:GravityField, ):
    G = 0.0000000000667; 
    xs, ys, zs = gf.field;
    pointmasses:PointMass = gf.pointmasses

    forces = {}
    counter = 0
    for point in pointmasses:
        xd = xs - point.x
        yd = ys - point.y
        zd = zs - point.z

        dimlist = []
        for dimension in [xd,yd,zd]:
            dimension_sign = -(dimension/np.abs(dimension))
            fg_dimension = dimension_sign*(G*point.mass)*(1/np.square(dimension))
            dimlist.append(fg_dimension)

        counter +=1

        forces[counter] = dimlist

    return forces

gf = GravityField(make_model_field(100,100,100), pointmasses=[
    PointMass(50,50,50,100000),
    # PointMass(2,2,2, 20000),
    # PointMass(8,3,3,800000)
    # PointMass(8,8,1,50000),
    # PointMass(2, 1, 9, 50000)
])

forces = evaluate_gravity_force(gf)

def sum_forces(forcedict:dict):
    xs = [dim[0] for dim in forcedict.values()]
    ys = [dim[1] for dim in forcedict.values()]
    zs = [dim[2] for dim in forcedict.values()]

    out_x = np.nansum(xs, axis=0)
    out_y = np.nansum(ys, axis=0)
    out_z = np.nansum(zs, axis=0)

    return(out_x, out_y, out_z)


test = sum_forces(forces)


def make_distance_array(list_of_points: list[PointMass]):
    # preallocate the array with shape = number of points * number of points
    base_arr = np.zeros((len(list_of_points), len(list_of_points)))

    # now populate the array; row i, col j will be the distance between point i and point j

    for i in range(len(list_of_points)):
        for j in range(len(list_of_points)):
            # evaluate the distance formula. the i, j structure compares all points to each other and creates an array.
            # this method isn't efficient for large numbers of points; for large numbers of points only the lower (or upper) diagonal distance should be assessed because the distance is symmetric. 
            base_arr[i,j] = np.sqrt((list_of_points[i].x - list_of_points[j].x)**2 + (list_of_points[i].y - list_of_points[j].y)**2 + (list_of_points[i].z - list_of_points[j].z)**2)

    return base_arr

#test for make_distance_array:
assert(make_distance_array([PointMass(0,0,0,1,0), PointMass(1,1,1,1,0)])[0,1] == 1.7320508075688772, "Distance formula is not correct")

#test that length and width are correct for the number of pointmasses passed:

assert(len(make_distance_array([PointMass(0,0,0,1,0), PointMass(1,1,1,1,0)])) == 2, "Length of distance array is not correct")

#test that the distance array is symmetric, i.e. that i,j = j,i

a1 = make_distance_array([PointMass(0,0,0,0,0), PointMass(1,1,1,1,0)])
assert(a1[0,1] == a1[1,0], "Distance array is not symmetric")

a2 = make_distance_array([PointMass(0,0,0,0,0), PointMass(1,1,1,1,0), PointMass(2,2,2,2,0), PointMass(3,3,3,3,0), PointMass(4,4,4,4,0)])


#for the numba version we need to fix the preprocess a bit for typing considerations

list_of_points_numba = [(x,x,x, np.random.rand()*1000000) for x in range(100)] #just a set of tuples to handle

@numba.jit(parallel=True)
def make_distance_array_numba(list_of_points: list[tuple[float, float, float, float]]):
    # preallocate the array with shape = number of points * number of points
    base_arr = np.zeros((len(list_of_points), len(list_of_points)))

    # now populate the array; row i, col j will be the distance between point i and point j

    for i in range(len(list_of_points)):
        for j in range(len(list_of_points)):
            # extract the position components only, ignoring the mass
            xi, yi, zi, _ = list_of_points[i]
            xj, yj, zj, _ = list_of_points[j]

            base_arr[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

    return base_arr

#same tests for the numba version:

start = time.perf_counter()
distances = make_distance_array_numba(list_of_points_numba)
end= time.perf_counter()
print(f"Array completeion time: {end-start} seconds for {len(list_of_points_numba)} points")

@numba.jit(parallel=True)
def make_mass_array_numba(list_of_points: list[tuple[float, float, float, float, tuple[float,float,float]]]):
    # preallocate the array with shape = number of points * number of points
    base_arr = np.zeros((len(list_of_points), len(list_of_points)))

    # row i, col j is the product of m1 and m2
    for i in range(len(list_of_points)):
        for j in range(len(list_of_points)):
            base_arr[i, j] = list_of_points[i][3] * list_of_points[j][3] # this is the mass of the interacting force pair

    return base_arr

start= time.perf_counter()
masses = make_mass_array_numba(list_of_points_numba)
end= time.perf_counter()
print(f"Mass array completeion time: {end-start} seconds for {len(list_of_points_numba)} points")

#eval force of gravity between each set of points by dividing the mass product array by the distance array squared for each position. 

@numba.jit(parallel=True)
def evaluate_gravity_force_numba(mass_array:np.ndarray, distance_array:np.ndarray):
    G = 0.0000000000667;
    return ((G*mass_array)/(distance_array**2))

start= time.perf_counter()
evaluate_gravity_force_numba(masses, distances)
end= time.perf_counter()
print(f"Force array completeion time: {end-start} seconds for {len(list_of_points_numba)} points")

#now write a CUDA version of the same array to pass on to the GPU

@cuda.jit
def evaluate_gravity_force_cuda(mass_array:np.ndarray, distance_array:np.ndarray, force_array:np.ndarray): 
    G = 0.0000000000667;
    x,y = numba.cuda.grid(2)
    if x < mass_array.shape[0] and y < mass_array.shape[1]:
        force_array[x,y] = ((G*mass_array[x,y])/(distance_array[x,y]**2))
    

force_array = np.zeros(masses.shape)
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(masses.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(masses.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
start = time.perf_counter()
evaluate_gravity_force_cuda[blockspergrid, threadsperblock](masses, distances, force_array)
end = time.perf_counter()
print(f"Force array CUDA completeion time: {end-start} seconds for {len(list_of_points_numba)} points")

#now implement a complete example of calling the above functions; this works for one step of a simulation

#initialize a set of 1000 random points with random masses:
pointlist = [(np.random.rand()*1000, np.random.rand()*1000, np.random.rand()*1000, np.random.rand()*1000000) for _ in range(400)]
distances = make_distance_array_numba(pointlist)
masses = make_mass_array_numba(pointlist)
forces = np.zeros(masses.shape)
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(masses.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(masses.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
start = time.perf_counter()
evaluate_gravity_force_cuda[blockspergrid, threadsperblock](masses, distances, forces)
end = time.perf_counter()
print(f"Force array CUDA completeion time: {end-start} seconds for {len(pointlist)} points")

#reduce the forces returned from this operation to net forces on each point

#this entails going from a specification ITO force-pairs to one where each point's XYZ magnitude is calculated. So the vector rhat from each point needs an XYZ basis.

@numba.jit(parallel=True)
def make_3d_distancearray(list_of_points: list[tuple[float, float, float, float, tuple[float,float,float]]]):
    # preallocate the array with shape = number of points * number of points
    base_arr_x = np.zeros((len(list_of_points), len(list_of_points)))
    base_arr_y = np.zeros((len(list_of_points), len(list_of_points)))
    base_arr_z = np.zeros((len(list_of_points), len(list_of_points)))

    # now populate the array; row i, col j will be the distance between point i and point j

    for i in range(len(list_of_points)):
        for j in range(len(list_of_points)):
            # extract the position components only, ignoring the mass
            xi, yi, zi, _, _ = list_of_points[i]
            xj, yj, zj, _, _ = list_of_points[j]

            base_arr_x[i,j] = xi - xj
            base_arr_y[i,j] = yi - yj
            base_arr_z[i,j] = zi - zj


    return (base_arr_x, base_arr_y, base_arr_z)


pointlist = [(np.random.rand()*1000, np.random.rand()*1000, np.random.rand()*1000, np.random.rand()*1000000,(0,0,0)) for _ in range(400)]

@numba.jit(parallel=True)
def update_pointlist(pointlist: list[tuple[float, float, float, float, tuple[float,float,float]]], outforces: list[np.ndarray], timestep: float):
    for i in range(len(pointlist)):
        pointlist[i] = (
            pointlist[i][0] + pointlist[i][4][0]*timestep + 0.5*((outforces[0][i]/pointlist[i][3])*timestep**2), # new x position is the old x position + the velocity and acceleration-related stuff
            pointlist[i][1] + pointlist[i][4][1]*timestep + 0.5*((outforces[1][i]/pointlist[i][3])*timestep**2), # new y position is the old y position + the velocity and acceleration-related stuff
            pointlist[i][2] + pointlist[i][4][2]*timestep + 0.5*((outforces[2][i]/pointlist[i][3])*timestep**2), # new z position is the old z position + the velocity and acceleration-related stuff
            pointlist[i][3], # mass is the same every time
            (pointlist[i][4][0] + (outforces[0][i]/pointlist[i][3])*timestep, #subtuple here of updated velocities after using the previous vals for the computation
            pointlist[i][4][1] + (outforces[1][i]/pointlist[i][3])*timestep, 
            pointlist[i][4][2] + (outforces[2][i]/pointlist[i][3])*timestep,)
            ) 
        
    return pointlist

def evaluate_model_run(pointlist: list[tuple[float, float, float, float, tuple[float,float,float]]], steps:int, timestep:float):
    """
    @param pointlist: a list of tuples representing points in 3D space, each containing 5 elements - x, y, z, mass, and a tuple of velocities
    @param steps: an integer representing the number of steps to evaluate
    @param timestep: a float representing the time step size
    @return: a list of updated points after evaluating the model run
    """

    for _ in range(steps):
        dists = make_3d_distancearray(pointlist)
        masses = make_mass_array_numba(pointlist)
        signs = [np.sign(dist) for dist in dists]
        forceslist = [np.zeros(masses.shape), np.zeros(masses.shape), np.zeros(masses.shape)]

        for dimension, farray in zip(dists, forceslist):

            threadsperblock = (16, 16)

            blockspergrid_x = int(np.ceil(masses.shape[0] / threadsperblock[0]))
            blockspergrid_y = int(np.ceil(masses.shape[1] / threadsperblock[1]))

            blockspergrid = (blockspergrid_x, blockspergrid_y)

            evaluate_gravity_force_cuda[blockspergrid, threadsperblock](masses, dimension, farray)

        outforces = []
        for dim, s in zip(forceslist, signs):
            dim = np.tril(dim, k=-1) # take the lower triangle of each, since they're symmetric; this ignores the inf spots naturally
            dim = dim*s # get the sign of each force
            ptforces = np.nansum(dim, axis=0)
            outforces.append(ptforces)

        pointlist = update_pointlist(pointlist, outforces, timestep)

    return pointlist

start = time.perf_counter()
evaluate_model_run(pointlist, 1000, 0.1)
end = time.perf_counter()
print(f"time to evaluate model run: {end - start:0.4f} seconds")



