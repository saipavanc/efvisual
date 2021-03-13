import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import optimize
import sympy
plt.rcParams['figure.figsize'] = [12, 8]

class particle:
    def __init__(self, q, pos_func_string):
        self.q = q
        self.pos_f_str = pos_func_string # string used to define the trajectory, list of two strings
        
        
        pos_expression = [sympy.simplify(i) for i in self.pos_f_str]
        vel_expression = [i.diff("t") for i in pos_expression]
        acc_expression = [i.diff("t") for i in vel_expression]
        
        
        a, b, t, t_r = sympy.symbols("a, b, t, t_r")
        pos_expression_r = [i.subs(sympy.symbols("t"),sympy.symbols("t_r")) for i in pos_expression]

        t_r_exp = -(t_r - t)**2 + (a - pos_expression_r[0])**2 + (b - pos_expression_r[1])**2
        
            
        self.t_r = sympy.lambdify((a, b, t, t_r), t_r_exp, "numpy")
        self.x_f = sympy.lambdify(([sympy.symbols("t")]), pos_expression , "numpy")
        self.v_f = sympy.lambdify(([sympy.symbols("t")]), vel_expression , "numpy")
        self.a_f = sympy.lambdify(([sympy.symbols("t")]), acc_expression , "numpy")

class reference_frame:
    def __init__(self, ran, N):
        coords = np.linspace(-ran, ran, N)
        X, Y = np.meshgrid(coords, coords) # restricting the motion to 2D
        self.X = X
        self.Y = Y
        self.range = ran
        
        # Electric fields
        plain_array = np.zeros_like(X)
        self.Ex = plain_array
        self.Ey = plain_array
        self.Ez = plain_array
        
        # Magnetic fields
        self.Bx = plain_array
        self.By = plain_array
        self.Bz = plain_array
    
    def ret_time(self, p, t):
        t_r = np.zeros_like(self.X)
        for i in range(len(t_r)):
            for j in range(len(t_r)):

                def g(t_r,i,j):
                    return p.t_r(self.X, self.Y, t, t_r)[i, j]
                t_r[i, j]=optimize.fsolve(g, t-10, args=(i,j))
        return t_r
    
    def E_fields(self, p, t):
        """
        Returns the E, B and the Poynting vector of the system
        """
        x_p = p.x_f(t) #present position
        
        t_r = self.ret_time(p, t)

        x_r = p.x_f(t_r)
        v_r = p.v_f(t_r)
        a_r = p.a_f(t_r)
        
        R = ((self.X - x_r[0])**2 + (self.Y - x_r[1])**2)**0.5
        
        n_x = np.divide((self.X - x_r[0]), R, out=np.zeros_like(t_r), where=t_r!=0) # vector n
        n_y = np.divide((self.Y - x_r[1]), R, out=np.zeros_like(t_r), where=t_r!=0)
        
        γ_r = 1/(1-(v_r[0]**2 + v_r[1]**2))**0.5 # gamma at retarded time
        
        rel_field_mag = np.divide(p.q/(γ_r**2*(1-(n_x*v_r[0] + n_y*v_r[1]))**3), R**2,out=np.zeros_like(t_r), where=R!=0)
        normal_field_mag = rel_field_mag * γ_r**2 * R
        
        E_x = rel_field_mag*(n_x - v_r[0]) + normal_field_mag*(n_y*a_r[1]*(n_x - v_r[0]) - n_y*a_r[0]*(n_y - v_r[1]))
        E_y = rel_field_mag*(n_y - v_r[1]) + normal_field_mag*(-n_x*a_r[1]*(n_x - v_r[0]) + n_x*a_r[0]*(n_y - v_r[1]))
        
        return (np.array([E_x, E_y]))
    
    def anim_video(self, p, interval, file_name):
        from matplotlib import animation  
        #constructing the first frame
        E = self.E_fields(p, 0)
        lt = 100

        fig = plt.figure()

        # marking the x-axis and y-axis 
        axis = plt.axes(xlim =(-self.range, self.range),  
                        ylim =(-self.range, self.range))  
        axis.set_aspect('equal', 'box')

        # initializing a line variable 
        Q = axis.quiver(self.X, self.Y,np.clip(E[0],-lt,lt),np.clip(E[1],-lt,lt))  
        [x,y] = p.x_f(0)
        l = axis.scatter(x,y)
        # data which the line will  
        # contain (x, y) 
        def update_plot(t, Q, l, self, p):
            [x,y] = p.x_f(t)
            l.set_offsets(np.array([x,y])) #plotting the charge position

            E = self.E_fields(p, t)
            Q.set_UVC(np.clip(E[0],-lt,lt),np.clip(E[1],-lt,lt)) #plotting the fields
            return Q,


        anim = animation.FuncAnimation(fig, update_plot, fargs=(Q, l, self, p), blit=True, frames=np.linspace(0, interval, 24))
        anim.save(file_name + ".mp4", writer='ffmpeg', fps=4)
