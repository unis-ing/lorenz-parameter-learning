sigma = 10; rho = 28;  beta = 8/3; % Lorenz parameters
sigma_DA = 0.8*sigma; rho_DA = 0.8*rho; beta_DA = 0.8*beta; % DA parameters
t0 = 0; tf = 150; % Initial and final times
dt = 0.0001; % Time step % Note: ode45 predicts min(dt)=0.00193
dt_obs = 0.05; % How often solution is observed
dt_param = 2.0; % How often to update the parameters
p_tol = 0.0001; % Tolerence for parameter switching
mu = 1.8/dt; % AOT nudging parameter
mu_p = mu/1000; % mu for updating the parameters
eta = 1e-5; % Amplitude of noise of measurements
epsilon = 1e-5; % Stochastic forcing amplitude on Lorenz

t = t0:dt:tf; N = length(t); rng(0);
U = zeros(3,N); V = zeros(3,N);
U(:,1)=[8.15641407246436;10.8938717856828;22.3338694390332];V(:,1)=[0;0;0];
lorenz    = @(U,sto)[   sigma*(U(2)-U(1));U(1)*(rho   -U(3))-U(2);...
                     U(1)*U(2)-beta*U(3)] + sto;
lorenz_DA = @(V,FC) [sigma_DA*(V(2)-V(1));V(1)*(rho_DA-V(3))-V(2);...
                     V(1)*V(2)-beta_DA*V(3)] - FC;

e_sol=zeros(1,N); e_sigma=zeros(1,N); e_rho=zeros(1,N); e_beta=zeros(1,N);
obs_int = round(dt_obs/dt); param_int = round(dt_param/dt);
for ti = 1:(N-1)
    e_sol(ti) = norm(U(:,ti)-V(:,ti)); e_sigma(ti) = abs(sigma_DA-sigma);
    e_rho(ti) = abs(rho_DA-rho);        e_beta(ti) = abs(beta_DA-beta);
    if mod(ti,obs_int) == 1 % Update feedback control term
        FC = mu*(V(:,ti) - (U(:,ti) + eta*randn(3,1)));
    else
        FC = [0;0;0];
    end
    if (mod(ti,param_int)==0)
        if (abs(V(2,ti)-V(1,ti)) > p_tol)
            sigma_DA = sigma_DA - mu_p*(V(1,ti)-U(1,ti))/(V(2,ti)-V(1,ti));
        end
        if (abs(V(1,ti)) > p_tol)
            rho_DA   = rho_DA   - mu_p*(V(2,ti)-U(2,ti))/(V(1,ti));
        end
        if (abs(V(3,ti)) > p_tol)
            beta_DA  = beta_DA  + mu_p*(V(3,ti)-U(3,ti))/(V(3,ti));
        end
        lorenz_DA=@(V,FC)[sigma_DA*(V(2)-V(1));V(1)*(rho_DA-V(3))-V(2);...
                          V(1)*V(2)-beta_DA*V(3)] - FC;
    end
    U(:,ti+1) = U(:,ti) + dt*lorenz(U(:,ti),epsilon*sqrt(dt)*randn(3,1));
    V(:,ti+1) = V(:,ti) + dt*lorenz_DA(V(:,ti),FC);
end
semilogy(t,e_sol); hold on; semilogy(t,e_sigma); semilogy(t,e_rho); 
semilogy(t,e_beta); xlabel('time'); ylabel('Error'); axis('tight');
legend('Solution Error','|\Delta\sigma|','|\Delta\rho|','|\Delta\beta|');