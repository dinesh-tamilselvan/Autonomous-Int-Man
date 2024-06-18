# impemented using bisection method
import data_file
import functions
import copy
import logging
import time
import numpy as np
 

buff_c = 0

def state_esti(veh, u_allow_min, u_allow_max, temp_ind, t_init, feas_stat, sig_traj,d):   # tr_sim_num, sim_num, train_iter_num, learn_flag,, set, green_zone):

    X = []
    ##### linear mapping #####
    if u_allow_max != None and u_allow_min != None:
    
        if u_allow_max>veh.u_max or u_allow_min<veh.u_min or u_allow_max<veh.u_min or u_allow_min>veh.u_max: 
            print("in state esti","u_max",u_allow_max,"u_min",u_allow_min)
            print(fdgfdgdfgdf)
        
        ################## override ###############
        ####veh.alpha = 0.5#np.random.random(1)[0]   #1
        assert 0<=veh.alpha <=1,f"value: {veh.alpha}"
        #print(veh.alpha)

        #print(f'value: {veh.ovr_stat[t_init]}')
        #print(f'**********over_ride:  {veh.ovr_stat[t_init]},{len(veh.ovr_stat[t_init])}')
        if veh.lane in veh.ovr_stat[t_init]:
            if veh.id in veh.ovr_stat[t_init][veh.lane]:veh.alpha = 1
        veh.u_traj.append((veh.alpha * (u_allow_max - u_allow_min)) + u_allow_min)


        #print("start values-1",veh.t_ser,veh.p_traj,veh.v_traj,veh.u_traj)
        #print("start values-1",type(veh.t_ser),type(veh.p_traj),type(veh.v_traj),type(veh.u_traj))

        #assert veh.u_traj[-1].isnumeric()
        ##### linear mapping #####
        X.append(veh.p_traj[temp_ind])
        X.append(veh.v_traj[temp_ind])
        x_next = functions.compute_state(X, veh.u_traj[temp_ind],round( data_file.dt,1),veh.v_max,0)


        if x_next[1]>veh.v_max or x_next[1]<veh.v_min:
            assert False, f'clipping not working, veh_vmax:{veh.v_max}, veh_vmin :{veh.v_min} pos:{x_next[0]}, vel:{x_next[1]}, pos_vec:{veh.p_traj}, vel_vec:{veh.v_traj}, sig:{sig_traj}' 
            
            ###### clipping code  ########

            #if x_next[1]>veh.v_max:
            #    delta = (veh.v_max - veh.v_traj[-1])/veh.u_traj[-1] #round((veh.v_max - veh.v_traj[-1])/veh.u_traj[-1],1)
            #    v_prime =  round(veh.v_max,1)
            #elif x_next[1]<veh.v_min:
            #    delta = (veh.v_min - veh.v_traj[-1])/veh.u_traj[-1] #round((veh.v_min - veh.v_traj[-1])/veh.u_traj[-1],1)
            #    v_prime =  round(veh.v_min,1)
            #if  delta<0 or delta>1:
            #    print("DELTA-------",delta,x_next[1] )
            #    exit()
            #p_prime =  veh.p_traj[-1] + veh.v_traj[-1]*delta + 0.5*veh.u_traj[-1]*(delta**2)
            #u_prime = 0

            #### X_next
            #p_next1 =  p_prime + v_prime*(round(data_file.dt,1) - delta) 
            #v_next1 =  v_prime 
            
            p_next1 =  x_next[0]
            v_next1 =  x_next[1]
            if round(p_next1, 4) > 0: assert sig_traj=='G' or (sig_traj=='R' and feas_stat == False), f"u_min: {u_allow_min}, u_max: {u_allow_max}, alpha: {veh.alpha}, veh.u: {(veh.alpha * (u_allow_max - u_allow_min) )+ u_allow_min},\
                ptraj: {veh.p_traj}\nvtraj: {veh.v_traj}, utraj: {veh.u_traj},set: {sig_traj}, ID:{veh.id}, sig_traj :{sig_traj}, feas_state:{feas_stat}, p_next:{p_next1}, v_pred:{ x_next[1]}, v_esti:{v_next1}, D_val:{d}, sig_traj:{veh.global_sig} "


            veh.p_traj.append(copy.deepcopy(p_next1))
            veh.v_traj.append(copy.deepcopy(v_next1))
            veh.t_ser.append(round((t_init + (round(data_file.dt,1))), 1))
        elif x_next[1]<=veh.v_max or x_next[1]>=veh.v_min:
        #elif veh.v_traj[-1]<=1.5 or veh.v_traj[-1]>=0:
        #elif veh.v_traj[-1]<=veh.v_max or veh.v_traj[-1]>=veh.v_min:    
        
            if round(x_next[0], 4) > 0: assert sig_traj=='G' or (sig_traj=='R' and feas_stat == False),f" stat:{feas_stat} sig:{sig_traj}   u_min: {u_allow_min}, u_max: {u_allow_max}, alpha: {veh.alpha}, veh.u: {(veh.alpha * (u_allow_max - u_allow_min) )+ u_allow_min},\
                ptraj: {veh.p_traj}\n vtraj: {veh.v_traj}, utraj: {veh.u_traj}, pos_next:{x_next[0]},v_esti:{x_next[1]}, D_val:{d}  set: {set}, ID:{veh.id}, sig_traj:{veh.global_sig} "

            veh.p_traj.append(copy.deepcopy(x_next[0]))
            veh.v_traj.append(copy.deepcopy(x_next[1]))
            veh.t_ser.append(round((t_init + (data_file.dt)), 1))

        veh.u_safe_min[veh.t_ser[-2]] = u_allow_min
        veh.u_safe_max[veh.t_ser[-2]] = u_allow_max
        
        veh.finptraj[veh.t_ser[-1]] = veh.p_traj[-1]
        veh.finvtraj[veh.t_ser[-1]] = veh.v_traj[-1]
        veh.finutraj[veh.t_ser[-2]] = veh.u_traj[-1]   


        #print(f't_ser:{veh.t_ser}')
        temp_ind = functions.find_index(veh, t_init)
        #print(f"cuur id: {veh.id}, current time: {t_init}")
        #print(f"cur pos: {veh.p_traj}, vel: {veh.v_traj},acc: {veh.u_traj}, t: {veh.t_ser}")

        assert (veh.t_ser[temp_ind+1])== t_init + (round(data_file.dt,1))
    elif  u_allow_max == None and u_allow_min == None: pass
    else: print(stoooop)        
    

    return veh, feas_stat    


def green_map(veh, _pre_v, t_init):#, sim_num, train_iter_num, tr_sim_num, learn_flag,set=None):


    feas_stat = True
    d = None
    if _pre_v!=None:
        pre_ind = functions.find_index(_pre_v, t_init)
        pre_ind_nxt = functions.find_index(_pre_v, t_init+round(data_file.dt,1))
        if ((pre_ind == None) or (_pre_v.p_traj[pre_ind] > (_pre_v.intsize + _pre_v.length - _pre_v.int_start))) and \
              ((pre_ind_nxt == None) or (_pre_v.p_traj[pre_ind_nxt] > (_pre_v.intsize + _pre_v.length - _pre_v.int_start))):
            _pre_v = None

    if _pre_v == None:
        if len(veh.t_ser) < 1:
            veh.t_ser = [veh.sp_t]
            veh.p_traj = [veh.p0]
            veh.v_traj = [veh.v0]
            veh.u_traj = []

        if len(veh.p_traj) >=1:
            temp_ind = functions.find_index(veh, t_init)
            veh.t_ser = veh.t_ser[:temp_ind+1]
            veh.p_traj = veh.p_traj[:temp_ind+1]
            veh.v_traj = veh.v_traj[:temp_ind+1]
            veh.u_traj = veh.u_traj[:temp_ind]
            #u_min_post = max(veh.u_min, -((veh.v_traj[temp_ind])/round( data_file.dt,1))) ## so vehicle never gets negative velocity
            #assert u_min_post >= veh.u_min
            #u_til = (veh.v_max - veh.v_traj[temp_ind]) /round( data_file.dt,1)
            #assert u_til <= veh.u_max
 

            u_allow_max = veh.u_max
            u_allow_min = veh.u_min #u_min_post
            assert u_allow_max >= u_allow_min
            

    elif _pre_v != None:
        if len(veh.t_ser) < 1:
            veh.t_ser = [veh.sp_t]
            veh.p_traj = [veh.p0]
            veh.v_traj = [veh.v0]
            veh.u_traj = []
            
        if len(veh.p_traj) >=1:   
            temp_ind = functions.find_index(veh, t_init)
            pre_ind = functions.find_index(_pre_v, t_init)
            pre_ind_nxt = functions.find_index(_pre_v, (t_init+round(data_file.dt,1)))
            
            assert temp_ind!= None,'current veh not present'
            assert pre_ind!= None,'current veh not present'
            assert pre_ind_nxt!= None, f'current veh not present, time: {(t_init+round( data_file.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'

            veh.t_ser = veh.t_ser[:temp_ind+1]
            veh.p_traj = veh.p_traj[:temp_ind+1]
            veh.v_traj = veh.v_traj[:temp_ind+1]
            veh.u_traj = veh.u_traj[:temp_ind]
        


            ################# need to modify the state estimation
            ############################
            ############################
            X = []
            X.append(veh.p_traj[temp_ind])
            X.append(veh.v_traj[temp_ind])
            #print(veh.u_traj,temp_ind,veh.p_traj, veh.t_ser)
            #print(veh.u_traj[temp_ind])
            x_prime_z = functions.compute_state(X, veh.u_min,round( data_file.dt,1),veh.v_max,0)
            
            assert _pre_v.p_traj[pre_ind_nxt] -  x_prime_z[0] >= data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime_z[1]**2 )/(2*veh.u_min))) , \
                f'GREEN SIGNAL: \n prev_pos:{_pre_v.p_traj[pre_ind_nxt] }, prev_vel:{_pre_v.v_traj[pre_ind_nxt]},  curr_pos:{ x_prime_z[0]},curr_vel:{ x_prime_z[1]}, \n \
                    RHS:{data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime_z[1]**2 )/(2*veh.u_min)))}, \n \
                    prev-pos:{_pre_v.p_traj}, vel: {_pre_v.v_traj}, curr-pos;{veh.p_traj}, vel: {veh.v_traj},\n \
                        curr_id:{veh.id},prev_id:{_pre_v.id}, timne:{t_init}'
            
            #if x_prime[1]**2 > 2*x_prime[0]*veh.u_min: feas_stat = False
            #else: feas_stat = True
            
            u_prime = veh.u_max
            u1 = veh.u_min
            u2 = veh.u_max
            
            X = []
            X.append(veh.p_traj[temp_ind])
            X.append(veh.v_traj[temp_ind])
            x_prime = functions.compute_state(X, u_prime,round( data_file.dt,1),veh.v_max,0)
            if _pre_v.p_traj[pre_ind_nxt] -  x_prime[0] >= data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime[1]**2 )/(2*veh.u_min))) and \
                x_prime[1]**2 <= 2*x_prime[0]*veh.u_min: 
                    u2 = u_prime
                    u_allow_max = u2
            else: 
                u_prime = (u2 + u1)/2
                while round(abs(u1-u2),data_file.delt)>0:
                    X = []
                    X.append(veh.p_traj[temp_ind])
                    X.append(veh.v_traj[temp_ind])
                    x_prime = functions.compute_state(X, u_prime,round( data_file.dt,1),veh.v_max,0)
                    if _pre_v.p_traj[pre_ind_nxt] -  x_prime[0] >= data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime[1]**2 )/(2*veh.u_min))) and \
                        x_prime[1]**2 <= 2*x_prime[0]*veh.u_min: u1 =  u_prime 
                    else:  u2 =  u_prime 
                    u_prime = (u2 + u1)/2
                    assert u1<=u2
                u_allow_max = min(u1,u2)

            #u_allow_max = min(u1,u2)  
            u_allow_min = veh.u_min 
            assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
            
            if (u_allow_max - u_allow_min) <= 10**-4:
                    u_allow_max = u_allow_min

            ######## R-end safety constraint with diff Umin #############

    return state_esti(veh, u_allow_min, u_allow_max, temp_ind,t_init,feas_stat,'G',d) #tr_sim_num, sim_num, train_iter_num, learn_flag,,set,green_zone)       



def red_map(veh, _pre_v, t_init ):#, sim_num, train_iter_num, tr_sim_num, learn_flag,set=None):

    feas_stat = True
    d = None
    if _pre_v!=None:
        pre_ind = functions.find_index(_pre_v, t_init)
        pre_ind_nxt = functions.find_index(_pre_v, t_init+round(data_file.dt,1))
        if ((pre_ind == None) or (_pre_v.p_traj[pre_ind] > (_pre_v.intsize + _pre_v.length - _pre_v.int_start))) and \
            ((pre_ind_nxt == None) or (_pre_v.p_traj[pre_ind_nxt] > (_pre_v.intsize + _pre_v.length - _pre_v.int_start))):
            _pre_v = None


    if _pre_v == None:
        #print(veh.id)
        if len(veh.t_ser) < 1:
            veh.t_ser = [veh.sp_t]
            veh.p_traj = [veh.p0]
            veh.v_traj = [veh.v0]
            veh.u_traj = []

        if len(veh.p_traj) >=1:
            temp_ind = functions.find_index(veh, t_init)
            
            #print(f'{veh.t_ser},{temp_ind},{t_init},pos:{veh.p_traj},vel:{veh.v_traj},u:{veh.u_traj}')
            veh.t_ser = veh.t_ser[:temp_ind+1]
            veh.p_traj = veh.p_traj[:temp_ind+1]
            veh.v_traj = veh.v_traj[:temp_ind+1]
            veh.u_traj = veh.u_traj[:temp_ind]


            ############### IEP constraints ##################
            X = []
            X.append(veh.p_traj[temp_ind])
            X.append(veh.v_traj[temp_ind])
            x_prime_z = functions.compute_state(X, veh.u_min,round( data_file.dt,1),veh.v_max,0)
            #assert _pre_v.p_traj[pre_ind_nxt] -  x_prime_z[0] >= data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime_z[1]**2 )/(2*veh.u_min))), f'prev:{_pre_v.p_traj} '    # RES
            if x_prime_z[1]**2 <= 2*x_prime_z[0]*veh.u_min:
                
                feas_stat = True
                u_prime = veh.u_max
                u1 = veh.u_min
                u2 = veh.u_max
                
                X = []
                X.append(veh.p_traj[temp_ind])
                X.append(veh.v_traj[temp_ind])
                x_prime = functions.compute_state(X, u_prime,round( data_file.dt,1),veh.v_max,0)
                if x_prime[1]**2 <= 2*x_prime[0]*veh.u_min: 
                    u2 = u_prime
                    u_allow_max = u2 
                else: 
                    u_prime = (u2 + u1)/2
                    while round(abs(u1-u2),data_file.delt)>0:
                        X = []
                        X.append(veh.p_traj[temp_ind])
                        X.append(veh.v_traj[temp_ind])
                        x_prime = functions.compute_state(X,u_prime,round( data_file.dt,1),veh.v_max,0)
                        if x_prime[1]**2 <= 2*x_prime[0]*veh.u_min: u1 =  u_prime 
                        else:  u2 =  u_prime 
                        u_prime = (u2 + u1)/2
                        assert u1<=u2
                    u_allow_max = min(u1,u2) 
                        

                #u_allow_max = min(u1,u2)   
                u_allow_min = veh.u_min 
                assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
                if (u_allow_max - u_allow_min) <= 10**-4: u_allow_max = u_allow_min

            elif x_prime_z[1]**2 > 2*x_prime_z[0]*veh.u_min:
                feas_stat = False
                u_allow_max = None
                u_allow_min = None         


    elif _pre_v != None:
        if len(veh.t_ser) < 1:
            veh.t_ser = [veh.sp_t]
            veh.p_traj = [veh.p0]
            veh.v_traj = [veh.v0]
            veh.u_traj = []
            
        if len(veh.p_traj) >=1:   
            
            #print("inside red","id:",veh.id)
            temp_ind = functions.find_index(veh, t_init)
            pre_ind = functions.find_index(_pre_v, t_init)
            pre_ind_nxt = functions.find_index(_pre_v, (t_init+round( data_file.dt,1)))
            
            assert temp_ind!= None,'current veh not present'
            assert pre_ind!= None,'current veh not present'
            assert pre_ind_nxt!= None, f'current veh not present, time: {(t_init+round( data_file.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'
            veh.t_ser = veh.t_ser[:temp_ind+1]
            veh.p_traj = veh.p_traj[:temp_ind+1]
            veh.v_traj = veh.v_traj[:temp_ind+1]
            veh.u_traj = veh.u_traj[:temp_ind]

            ######## RES + IEP #############

            X = []
            X.append(veh.p_traj[temp_ind])
            X.append(veh.v_traj[temp_ind])
            x_prime_z = functions.compute_state(X, veh.u_min,round( data_file.dt,1),veh.v_max,0)
            assert _pre_v.p_traj[pre_ind_nxt] -  x_prime_z[0] >= data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime_z[1]**2 )/(2*veh.u_min))), \
                f'RED SIGNAL: \n prev_pos:{_pre_v.p_traj[pre_ind_nxt] }, prev_vel:{_pre_v.v_traj[pre_ind_nxt]},  curr_pos:{ x_prime_z[0]},curr_vel:{ x_prime_z[1]}, \
                RHS:{data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime_z[1]**2 )/(2*veh.u_min)))}, \n \
                    prev-pos:{_pre_v.p_traj}, prev_vel: {_pre_v.v_traj}, prev_time: {_pre_v.t_ser}, prev_utraj:{_pre_v.u_traj}, \n curr-pos:{veh.p_traj}, curr_vel: {veh.v_traj},\
                        prev_time: {veh.t_ser}, prev_utraj:{veh.u_traj},\n \
                        curr_id:{veh.id},prev_id:{_pre_v.id}, time:{t_init}'

            if x_prime_z[1]**2 <= 2*x_prime_z[0]*veh.u_min:
                feas_stat = True
                u_prime = veh.u_max
                u1 = veh.u_min
                u2 = veh.u_max
                
                X = []
                X.append(veh.p_traj[temp_ind])
                X.append(veh.v_traj[temp_ind])
                x_prime = functions.compute_state(X, u_prime,round( data_file.dt,1),veh.v_max,0)
                if _pre_v.p_traj[pre_ind_nxt] -  x_prime[0] >= data_file.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime[1]**2 )/(2*veh.u_min))) and \
                    x_prime[1]**2 <= 2*x_prime[0]*veh.u_min: 
                        u2 = u_prime
                        u_allow_max = u2 
                else: 
                    u_prime = (u2 + u1)/2
                    while round(abs(u1-u2),data_file.delt)>0:
                        X = []
                        X.append(veh.p_traj[temp_ind])
                        X.append(veh.v_traj[temp_ind])
                        x_prime = functions.compute_state(X, u_prime,round( data_file.dt,1),veh.v_max,0)
                        if _pre_v.p_traj[pre_ind_nxt] -  x_prime[0] >= data_file.L + max(0,((_pre_v.p_traj[pre_ind_nxt]**2 - x_prime[0]**2 )/2*veh.u_min)) and \
                            x_prime[1]**2 <= 2*x_prime[0]*veh.u_min: u1 =  u_prime 
                        else:  u2 =  u_prime 
                        u_prime = (u2 + u1)/2
                        assert u1<=u2
                    u_allow_max = min(u1,u2)    
                        #print(f'u1:{u1},u2:{u2}')

                #u_allow_max = min(u1,u2)  
                u_allow_min = veh.u_min 
                assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
                if (u_allow_max - u_allow_min) <= 10**-4: u_allow_max = u_allow_min

            elif x_prime_z[1]**2 > 2*x_prime_z[0]*veh.u_min:
                feas_stat = False
                u_allow_max = None
                u_allow_min = None         

                  
    #print(f"time:{t_init}, pred_id:{_pre_v.id},cur_id:{veh.id},A:{acoeff_re},B:{bcoeff_re},C:{ccoeff_re}, fmax:{acoeff_re*u_allow_max**2+ bcoeff_re*u_allow_max + ccoeff_re},fmin:{acoeff_re*u_allow_min**2+ bcoeff_re*u_allow_min + ccoeff_re} ----***")
    
    return state_esti(veh, u_allow_min, u_allow_max, temp_ind,t_init,feas_stat,'R',d)# tr_sim_num, sim_num, train_iter_num, learn_flag,,set,green_zone)       




########### what to add here from coordinate pahe ############
# fin_traj
##


#########################################






















