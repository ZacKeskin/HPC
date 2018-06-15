import numpy as np

kernels = """

    inline float float_one()
    {
        return 1.0f;
    }

    inline double double_one()
    {
        return 1.0;

    }

    __kernel void compute_grid_geometry(__global const ${realType} *vertices,
                                        __global const int* elements,
                                        __global ${realType} *integrationElements,
                                        __global ${realType}3 *jacobians,
                                        __global ${realType}3 *jacobianInverseTransposed,
                                        __global ${realType}3 *normals)
    {

        int gid = get_global_id(0);
        ${realType}3 corners[3];
        ${realType}3 jac[2];
        ${realType} ata[2][2];
        ${realType} ata_inv[2][2];
        ${realType}3 jac_inv_trans[2];
        ${realType} inv_det_ata;
        
        uint index;
        int i;
        
        for (i = 0; i  < 3; ++i)
        {
            index = elements[3 * gid + i];
            corners[i] = (${realType}3)(vertices[3 * index], vertices[3 * index + 1], vertices[3 * index + 2]);
        }


        jac[0] = corners[1] - corners[0];
        jac[1] = corners[2] - corners[0];

        ata[0][0] = dot(jac[0], jac[0]);
        ata[0][1] = dot(jac[0], jac[1]);
        ata[1][0] = ata[0][1];
        ata[1][1] = dot(jac[1], jac[1]);
        
        inv_det_ata = ${realType}_one() / (ata[0][0] *ata[1][1] - ata[0][1] * ata[1][0]);
        integrationElements[gid] = rsqrt(fabs(inv_det_ata));

        ata_inv[0][0] = ata[1][1] * inv_det_ata;
        ata_inv[0][1] = -ata[0][1] * inv_det_ata;
        ata_inv[1][0] = ata_inv[0][1];
        ata_inv[1][1] = ata[0][0] * inv_det_ata;
        
        jac_inv_trans[0] = jac[0] * ata_inv[0][0] + jac[1] * ata_inv[1][0];
        jac_inv_trans[1] = jac[0] * ata_inv[0][1] + jac[1] * ata_inv[1][1];

        normals[gid] = normalize(cross(jac[0], jac[1]));
        
        jacobians[2 * gid] = jac[0];
        jacobians[2 * gid + 1] = jac[1];
        
        jacobianInverseTransposed[2 * gid] = jac_inv_trans[0];
        jacobianInverseTransposed[2 * gid + 1] = jac_inv_trans[1];


    }

    """

class Grid(object):
    """A Grid class that can handle two dimensional flat and surface grids."""
    
    def __init__(self, vertices, elements, dtype='float64'):
        """
        Initialize a grid.
        
        Parameters
        ----------
        vertices: (N, 3) Array
            Vertices are stored in an (N, 3) array of floating point numbers.
        elements: (N, 3) Array
            Elements are stored in an (N, 3) array of type 'np.int32'.
        dtype: string
          dtype is either 'float64' or 'float32'. Internally, all structures and the
          vertices are converted to the format specified by 'dtype'. This is useful
          for OpenCL computations on GPUs that do not properly support double precision
          ('float64') types
        """
        
        import numpy as np
        import pyopencl as cl
        from mako.template import Template

        ocl_type = {'float32': 'float',
                    'float64': 'double'}[dtype]
        
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        code = Template(kernels).render(realType=ocl_type)
        prg = cl.Program(ctx, code).build()
        
        req = ['A', 'C']
        
        self._vertices = np.require(vertices, dtype=dtype,
                                   requirements=req)
        self._elements = np.require(elements, dtype='int32',
                                   requirements=req)
        
        number_of_elements = self._elements.shape[0]
        number_of_vertices = self._vertices.shape[0]
        
        device = {}
        
        device['vertices'] = cl.Buffer(
            ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
            hostbuf=self._vertices)
            
        device['elements'] = cl.Buffer(
            ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
            hostbuf=self._elements)
        
        device['int_elements'] = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE, 
            size=number_of_elements * np.dtype(dtype).itemsize)

        device['jacobians'] = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE, 
            size=8 * number_of_elements * np.dtype(dtype).itemsize)
        
        device['jac_inv_trans'] = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE, 
            size=8 * number_of_elements * np.dtype(dtype).itemsize)
        
        device['normals'] = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE, 
            size=4 * number_of_elements * np.dtype(dtype).itemsize)
        
        
        compute_grid_geometry = prg.compute_grid_geometry
        
        compute_grid_geometry(
            queue, (number_of_elements,),
            None,
            device['vertices'],
            device['elements'],
            device['int_elements'],
            device['jacobians'],
            device['jac_inv_trans'],
            device['normals'])
        
        self._device = device
        
        self._int_elements = cl.enqueue_map_buffer(
            queue, device['int_elements'],
            cl.map_flags.READ, 0, (number_of_elements,),
            dtype, is_blocking=True)[0]

        self._jacobians = cl.enqueue_map_buffer(
            queue, device['jacobians'],
            cl.map_flags.READ, 0, (2 * number_of_elements, 4),
            dtype, is_blocking=True)[0][:,:3].T

        self._jac_inv_trans = cl.enqueue_map_buffer(
            queue, device['jac_inv_trans'],
            cl.map_flags.READ, 0, (2 * number_of_elements, 4),
            dtype, is_blocking=True)[0][:,:3].T

        self._normals = cl.enqueue_map_buffer(
            queue, device['normals'],
            cl.map_flags.READ, 0, (number_of_elements, 4),
            dtype, is_blocking=True)[0][:,:3].T
    
    @property
    def vertices(self):
        """Return the vertices."""
        return self._vertices
        
    @property
    def elements(self):
        """Return the elemenets."""
        return self._elements
    
    def corners(self, element_index):
        """Return the corners of a given element as (3, 3) array"""
        return self._vertices[self._elements[element_index, :], :]
    
    def integration_element(self, element_index):
        """Return |det J| for a given element."""
        return self._int_elements[element_index]
    
    def normal(self, element_index):
        """Return the exterior pointing normal of an element."""
        return self._normals[:, element_index]
    
    def inverse_jacobian_transpose(self, element_index):
        """Return the (3, 3) inverse Jacobian transpose of an element."""
        return self._jac_inv_trans[:, 2 * element_index : 2 * element_index + 2]
    
    @classmethod
    def from_file(cls, file_name, dtype='float64'):
        """Read a mesh from a vtk file."""
        
        import os.path
        import vtk
        import numpy as np

        if not os.path.isfile(file_name):
            raise ValueError("File does not exist.")


        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()
        data = reader.GetOutput()
        point_data = data.GetPoints()
        number_of_points = point_data.GetNumberOfPoints()

        points = np.zeros((number_of_points, 3), dtype=dtype)

        for index in range(number_of_points):
            points[index, :] = point_data.GetPoint(index)

        number_of_cells = data.GetNumberOfCells()

        elements = []

        for index in range(number_of_cells):
            cell = data.GetCell(index)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                elements.append(
                    [cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)]
                )
        elements = np.array(elements, dtype=np.int32)

        return cls(points, elements, dtype=dtype)

    def local_stiffness(self, element_index):

        detJ = self.integration_element(element_index)
        J = self.inverse_jacobian_transpose(element_index)
        temp = np.zeros((3,3))
        temp[:,:-1] = J
        J = temp

        K_local = np.zeros(shape=(3,3))
        K_local[0][0] =     (J[0][0]**2   +   2 * (J[0][0]*J[0][1])   +   J[0][1]**2     +
                            J[1][0]**2   +   2 * (J[1][0]*J[1][1])   +   J[1][1]**2     +
                            J[2][0]**2   +   2 * (J[2][0]*J[2][1])   +   J[2][1]**2     )
        K_local[0][1] =     (-(J[0][0] * J[0][1] + J[0][1]**2)   +
                            -(J[1][0] * J[1][1] + J[1][1]**2)   +
                            -(J[2][0] * J[2][1] + J[2][1]**2)   )
        K_local[0][2] =     (-(J[0][0]**2 + J[0][1] * J[0][0])   +   
                            -(J[1][0]**2 + J[1][1] * J[1][0])   +   
                            -(J[2][0]**2 + J[2][1] * J[2][0])   ) 
        K_local[1][0] =     (-(J[0][0] * J[0][1] + J[0][1]**2)   +
                            -(J[1][0] * J[1][1] + J[1][1]**2)   +
                            -(J[2][0] * J[2][1] + J[2][1]**2)   )
        K_local[1][1] =     (J[0][1]**2  +   J[1][1]**2   +    J[2][1]**2)
        K_local[1][2] =     (J[0][1]*J[0][0] +   J[1][1]*J[1][0]   +    J[2][1]*J[2][0])
        K_local[2][0] =     (-(J[0][0]**2 + J[0][1] * J[0][0])   +   
                            -(J[1][0]**2 + J[1][1] * J[1][0])   +   
                            -(J[2][0]**2 + J[2][1] * J[2][0])   ) 
        K_local[2][1] =     (J[0][1]*J[0][0] +   J[1][1]*J[1][0]   +    J[2][1]*J[2][0]) 
        K_local[2][2] =     (J[0][0]**2  +   J[1][0]**2   +    J[2][0]**2)


        return (K_local* detJ)

if __name__ == "__main__":
    import os
    filepath = os.path.join(os.getcwd(),'lshape.vtk')
    mesh = Grid.from_file(filepath,dtype = 'float64')

    print(mesh.local_stiffness(500))
    