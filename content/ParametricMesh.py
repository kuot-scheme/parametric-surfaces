import numpy
from stl import mesh
import vtkplotlib as vpl

# (i,j)-th entry in verts is a point in the domain.
def domainGen(domX, domY, m, n):

    delta_x = (domX[1] - domX[0])/m
    delta_y = (domY[1] - domY[0])/n

    verts = [ [None]*(n+1) for _ in range(m+1) ]
    
    for i in range(m+1):
        for j in range(n+1):
            verts[i][j] = [float(domX[0] + i*delta_x), float(domY[0] + j*delta_y), float(0)]

    return verts

# F: parametrization F(u,v) = [x,y,z]
# dom = [[u1,u2], [v1,v2]] : Domain of F
# dim = [M,N] : number of subdivisions in u, v directions, respectively
# scale : scaling size of final mesh
# output: pystl mesh type
def paramMesh(F, dom, dim, scale):
    m = dim[0]
    n = dim[1]
    surf = mesh.Mesh(numpy.zeros(2*m*n, dtype = mesh.Mesh.dtype))
    dom = domainGen(dom[0], dom[1], m, n)

    tri = []
    for i in range(m):
        for j in range(n):
            p11 = [scale*a for a in F(dom[i][j])]
            p12 = [scale*a for a in F(dom[i][j+1])]
            p21 = [scale*a for a in F(dom[i+1][j])]
            p22 = [scale*a for a in F(dom[i+1][j+1])]
            
            tri.extend([
                  [p11, p21, p22],
                  [p11, p22, p12]
            ])

    surf.vectors = tri

    return surf

# [u1,u2] is the cyclic parameter (e.g. theta in cylindrical coordinates)
def paramMeshCylindrical(F, dom, dim, scale = 1):
    m = dim[0]
    n = dim[1]
    surf = mesh.Mesh(numpy.zeros(2*m*n, dtype = mesh.Mesh.dtype))
    dom = domainGen(dom[0], dom[1], m, n)

    tri = []

    for i in range(m):
        for j in range(n):
            p11 = [scale*a for a in F(dom[i%m][j])]
            p12 = [scale*a for a in F(dom[i%m][j+1])]
            p21 = [scale*a for a in F(dom[(i+1)%m][j])]
            p22 = [scale*a for a in F(dom[(i+1)%m][j+1])]

            tri.extend([
                  [p11, p21, p22],
                  [p11, p22, p12]
            ])

    surf.vectors = tri

    return surf

# Solid bounded by the xy-plane and the graph z = F(x,y)
def graphMeshBase(F, dom, dim, scale = 1):
    m = dim[0]
    n = dim[1]
    surf = mesh.Mesh(numpy.zeros(4*m*n + 4*m + 4*n, dtype = mesh.Mesh.dtype))
    dom = domainGen(dom[0], dom[1], m, n)

    tri = []

    # top and base
    for i in range(m):
        for j in range(n):
            
            p11 = dom[i][j]
            p12 = dom[i][j+1]
            p21 = dom[i+1][j]
            p22 = dom[i+1][j+1]

            q11 = [scale*a for a in F(p11)]
            q12 = [scale*a for a in F(p12)]
            q21 = [scale*a for a in F(p21)]
            q22 = [scale*a for a in F(p22)]
            
            r11 = [scale*a for a in p11]
            r12 = [scale*a for a in p12]
            r21 = [scale*a for a in p21]
            r22 = [scale*a for a in p22]

            tri.extend([
                [q11, q21, q22],
                [q11, q22, q12],
                [r11, r21, r22],
                [r11, r22, r12]
            ])
    
    #sides
    for i in range(m):
        p11 = [scale*a for a in dom[i][0]]
        p12 = [scale*a for a in dom[i+1][0]]
        p21 = [scale*a for a in F(dom[i][0])]
        p22 = [scale*a for a in F(dom[i+1][0])]

        q11 = [scale*a for a in dom[i][n]]
        q12 = [scale*a for a in dom[i+1][n]]
        q21 = [scale*a for a in F(dom[i][n])]
        q22 = [scale*a for a in F(dom[i+1][n])]
        
        tri.extend([
            [p11, p21, p22],
            [p11, p22, p12],
            [q11, q21, q22],
            [q11, q22, q12]
        ])

    for i in range(n):
        p11 = [scale*a for a in dom[0][i]]
        p12 = [scale*a for a in dom[0][i+1]]
        p21 = [scale*a for a in F(dom[0][i])]
        p22 = [scale*a for a in F(dom[0][i+1])]

        q11 = [scale*a for a in dom[m][i]]
        q12 = [scale*a for a in dom[m][i+1]]
        q21 = [scale*a for a in F(dom[m][i])]
        q22 = [scale*a for a in F(dom[m][i+1])]
        
        tri.extend([
            [p11, p21, p22],
            [p11, p22, p12],
            [q11, q21, q22],
            [q11, q22, q12]
        ])

    surf.vectors = tri
    
    return surf

# Meshes showing grid lines
def paramMeshCylindricalGrid(F, dom, m, n, P, Q, r, s, scale = 1):

    p = int(numpy.ceil(r*P))
    q = int(numpy.ceil(s*Q))
    
    mm = m*P
    nn = n*Q + q
    
    domPoints = domainGen(dom[0], dom[1], mm, nn)
    tri = []

    for k in range(m):
        for i in range(p):
            for j in range(n*Q + q):
                p11 = domPoints[(k*P + i)%mm][j]
                p12 = domPoints[(k*P + i)%mm][j + 1]
                p21 = domPoints[(k*P + i + 1)%mm][j]
                p22 = domPoints[(k*P + i + 1)%mm][j + 1]

                q11 = [scale*a for a in F(p11)]
                q12 = [scale*a for a in F(p12)]
                q21 = [scale*a for a in F(p21)]
                q22 = [scale*a for a in F(p22)]

                tri.extend([
                  [q11, q21, q22],
                  [q11, q22, q12]
                ])
    
    for k in range(m + 1):
        for l in range(n + 1):
            for i in range(P - p):
                for j in range(q):
                    p11 = domPoints[(k*P + i + p)%mm][l*Q + j]
                    p12 = domPoints[(k*P + i + p)%mm][l*Q + j + 1]
                    p21 = domPoints[(k*P + i + p + 1)%mm][l*Q + j]
                    p22 = domPoints[(k*P + i + p + 1)%mm][l*Q + j + 1]
    
                    q11 = [scale*a for a in F(p11)]
                    q12 = [scale*a for a in F(p12)]
                    q21 = [scale*a for a in F(p21)]
                    q22 = [scale*a for a in F(p22)]
    
                    tri.extend([
                      [q11, q21, q22],
                      [q11, q22, q12]
                    ])
    
    surf = mesh.Mesh(numpy.zeros(len(tri), dtype = mesh.Mesh.dtype))
    surf.vectors = numpy.array(tri)

    return surf

def paramMeshGrid(F, dom, m, n, P, Q, r , scale = 1):

    p = int(numpy.ceil(r*P))
    q = int(numpy.ceil(r*Q))
    mm = m*P + p
    nn = n*Q + q

    domPoints = domainGen(dom[0], dom[1], mm, nn)
    tri = []

    for k in range(m + 1):
        for i in range(p):
            for j in range(n*Q + q):
                p11 = domPoints[k*P + i][j]
                p12 = domPoints[k*P + i][j + 1]
                p21 = domPoints[k*P + i + 1][j]
                p22 = domPoints[k*P + i + 1][j + 1]

                q11 = [scale*a for a in F(p11)]
                q12 = [scale*a for a in F(p12)]
                q21 = [scale*a for a in F(p21)]
                q22 = [scale*a for a in F(p22)]

                tri.extend([
                  [q11, q21, q22],
                  [q11, q22, q12]
                ])
    
    for k in range(m):
        for l in range(n + 1):
            for i in range(P - p):
                for j in range(q):
                    p11 = domPoints[k*P + i + p][l*Q + j]
                    p12 = domPoints[k*P + i + p][l*Q + j + 1]
                    p21 = domPoints[k*P + i + 1 + p][l*Q + j]
                    p22 = domPoints[k*P + i + 1 + p][l*Q + j + 1]
    
                    q11 = [scale*a for a in F(p11)]
                    q12 = [scale*a for a in F(p12)]
                    q21 = [scale*a for a in F(p21)]
                    q22 = [scale*a for a in F(p22)]
    
                    tri.extend([
                      [q11, q21, q22],
                      [q11, q22, q12]
                    ])
    
    surf = mesh.Mesh(numpy.zeros(len(tri), dtype = mesh.Mesh.dtype))
    surf.vectors = numpy.array(tri)

    return surf

# Combines a list of meshes into a single stl
def combine(meshes):
    return mesh.Mesh(numpy.concatenate([m.data for m in meshes]))

#plots mesh using vtkplotlib
def plotMesh(surf):

    figure = vpl.figure("Parametric surface")

    figure.background_color = [0.1,0.1,0.1]

    vpl.mesh_plot(surf)

    vpl.show()
