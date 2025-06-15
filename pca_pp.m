function [p,t,model] = pca_pp(xcs,pcs)

% Principal Component Analysis based on the eigendecompostion of XX.
%
% p = pca_pp(xcs)     % minimum call
% [p,t,model] = pca_pp(xcs,pcs)     % complete call
%
%
% INPUTS:
%
% xcs: [NxM] preprocessed billinear data set 
%
% pcs: [1xA] Principal Components considered (e.g. pcs = 1:2 selects the
%   first two PCs). By default, pcs = 0:min(size(xcs))
%
%
% OUTPUTS:
%
% p: [MxA] matrix of loadings.
%
% t: [NxA] matrix of scores.
%
% model: structure that contains model information.
%
%
% EXAMPLE OF USE: Random data:
%
% X = simuleMV(20,10,8);
% Xcs = preprocess2D(X,2);
% pcs = 1:3;
% [p,t] = pca_pp(Xcs,pcs);
%
%
% coded by: Jose Camacho Paez (josecamacho@ugr.es)
% last modification: 21/Apr/2023
%
% Copyright (C) 2023  University of Granada, Granada
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% Arguments checking

% Set default values
routine=dbstack;
assert (nargin >= 1, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
%Observaciones(FILAS)
N = size(xcs, 1);
%Variables(COLUMNAS)
M = size(xcs, 2);
if nargin < 2 || isempty(pcs), pcs = 0:rank(xcs); end;

% Convert column arrays to row arrays
if size(pcs,2) == 1, pcs = pcs'; end;

% Preprocessing
pcs = unique(pcs);
pcs(find(pcs==0)) = [];
pcs(find(pcs>size(xcs,2))) = [];
pcs(find(pcs>rank(xcs))) = [];
A = length(pcs);

% Validate dimensions of input data
assert (isequal(size(pcs), [1 A]), 'Dimension Error: 2nd argument must be 1-by-A. Type ''help %s'' for more info.', routine(1).name);

% Validate values of input data
assert (isempty(find(pcs<0)) && isequal(fix(pcs), pcs), 'Value Error: 2nd argument must contain positive integers. Type ''help %s'' for more info.', routine(1).name);


%% Main code

%Antes de ver los dos casos, debemos saber:

%Espacio de observaciones: Está definido por las filas de X
%(cada fila es una observación o muestra y estudiamos la relación entre 
% observaciones)

%Espacio de características: Está definido por las columnas de X
%(cada columna es una variable o característica y estudiamos la relación
% entre variables)


%% PRIMER CASO: N>M (más observaciones que variables)

%Calculamos la matriz de covarianza X^t*X (se nos queda una matriz
%de dimensión MxM) captura la relación entre las variables
%Obtenemos los autovalores y autovectores
% En este caso los autovectores de  X^t*X están en un espacio de dimensión M
% por lo que cada autovector son  componentes principales en el espacio de caracteristicas 
% El objetivo es buscar dichas componentes principales que capturen la
% máxima variabilidad.

if N>M,   %mas observaciones que variables
    XX = xcs'*xcs;
    [p,D] = eig(XX);
    [kk,ind] = sort(real(diag(D)),'descend');
    p = p(:,ind);
    t = xcs*p;

   
  %%SEGUNDO CASO  
    
    %Cuando hay más variables (M) que observaciones (N), la matriz
    %de la covarianza X^T*X puede volverse singular o no invertible
    %esto sucede porque la matriz tiene más dimensiones (varibales) 
    %que datos disponibles para definir relaciones entre ellas.

    %Si N>M (más observaciones que variables): 

    %La matriz X^T*X es de tamaño MxM, todos los autovalores son 
    %distintos de cero lo que permite obtener los autovectores
    %directamente en el espacio de caracteritiscas. 


    %Si N<M (más variables que observaciones)

    %La matriz X^T*X es de tamaño MxM, pero su rango máximo es N lo que
    %significa que al menos M-N autovalores serán ceros. Por ello en este
    %caso se calculo primero X*X^T (que es de tamaño NxN), obteniendo los
    %autovectores en espacio de observaciones y luego se proyectan en el
    %espacio de caracteristicas para obtener las componentes principales.
    %Buscamos autovectores en el espacio de características porque describen 
    % la relación entre variables (no cómo se parecen los individuos entre sí),
    % esto permite expresar los datos en una nueva base con menor redundancia y mayor interpretabilidad.
    


else,

    XX = xcs*xcs'; %matriz de covarianza de 
    %dimensión NxN. En este caso, trabajamos en 
    %espacio de observaciones en lugar del espacio
    %de características por lo que t representa 
    % directamente los scores.
    [t,D] = eig(XX);
    s = real(sqrt(real(diag(D)))); %obtenemos los valores singulares
    [kk,ind] = sort(s,'descend'); %ordenamos de manera descendente 
                                    %tanto los valores singulares como t
    t = t(:,ind).*(ones(N,1)*s(ind)'); %se ordenan los valores de t
    % y se escalan por su valor singular


    p = xcs'*t; %X^T*u = v. Esta ecuación nos dice que autovectores u de X*X^T (espacio
    %de observaciones) se tranforman en los autovectores v de X^T*X
    %(espacio de características) al aplicar X^T.

    %Demostración:

    %La descomposición en valores singulares (SVD)
    %nos indica que X =USV^T en donde:

    %U(NxN) contiene los autovectores de X*X^T
    %V(MxM) contiene los autovectores de X^T*X
    %S es la matriz diagonal de valores singulares,
    %son los autovalores de X*X^T y X^T*X


    %Multiplicamos X por su transpuesta: 
    %X^T*X = (U*S*V^T)^T * (U*S*V^T)
    %X^T * X = V*S*U^T*U*S*V^T
    %Como U es ortogonal se cumple: U^T*U = I
    %Por lo que se nos queda la siguiente ecuación
    %simplificada: X^T*X = V*S^2*V^T
    %Conclusiones: con esto podemos concluir que 
    %los autovectores de X^T*X están en V 



    %Si partimos de la ecuación original de la SVD
    %X = USV^T
    %Si multiplicamos por S^-1 (suponiendo que S es invertible en sus 
    %dimensiones válidas) en ambos lados: 
    % X * S^-1 = U * V^T 
    %Multiplicamos por U^T a la derecha: 
    % U^T * X* S^-1 = U^T * U *V^T
    % Se nos queda: U^T*X*S^-1 = V^T
    %Trasponiendo la ecuación se nos queda: V= X^T * U * S^-1
    %Esto nos demuestra que si tenemos los autovectores de U 
    % de X*X^T, podemos obtener los autovectores de X^T*X
    %multiplicando X^T por U y escalando S^-1.
    
    
    
    %Cuando aplicamos PCA, lo que queremos es encontrar un nuevo sistema
    %de coordenadas (pcs) que nos permita representar los datos originales
    %de manera más eficiente, estas nuevas direcciones deben estar
    %expresadas en términos de las variables originales (combinación lineal
    %de las variables orginales)por ello debemos de obtener la pcs en el espacio 
    % de las variables.

    for i=1:size(p,2)  %se normaliza dividiéndola por su norma euclidiana
        %para que cada autovector tenga norma 1 (ya que deben ser
        %ortonormales)
                      
        p(:,i) = p(:,i)/sqrt(p(:,i)'*p(:,i));
    end
end

%
p = p(:,pcs);
t = t(:,pcs);

model.var = trace(XX);
model.lvs = 1:size(p,2);
model.loads = p;
model.scores = t;
model.type = 'PCA';
        



