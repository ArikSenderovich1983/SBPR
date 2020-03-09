include "disjunctive.mzn";


%parameters
int: A;
int: R;
int: S;
set of int: Activities = 1..A;
set of int: Resources = 1..R;
set of int: Skills = 1..S;
array[Resources, Skills] of int: Sigma;
array[Resources, Activities] of int: weights;
array[Resources, Skills] of int: learning;

array[Resources, Activities] of int: social_cost;
array[Resources] of int: social_budget;
array[Activities] of int: S_a;
array[Skills, Skills] of int: prec;

%array[Resources, Skills] of int: quality;
%array[Resources] of int: quality_lower;


% Variables
array[Activities, Resources] of var 0..1: X; % assignment var
array[Skills, Resources] of var 0..1: Y; % assignment var

var int: Z; % wage objective variable
var int: L; % learning objective variable

% for every activity, we assign at most one resource type and every activity must be assigned
constraint forall(a in Activities)(sum(r in Resources)(X[a,r])==1); 
%social budget:
constraint forall(r in Resources)(sum(a in Activities)(social_cost[r,a]*X[a,r]) <= social_budget[r]); 
%constraint forall(r in Resources)(sum(s in Skills)(quality[r,s]*Y[s,r]) >= quality_lower[r]); 

%Skill learning constraints:
constraint forall(a in Activities, r in Resources)(((X[a,r])==1) /\ (Sigma[r,S_a[a]]==0) -> Y[S_a[a], r]==1); 
constraint forall(s in Skills, r in Resources)((Y[s,r]==1)->forall(v in Skills where prec[s,v]==1)(Y[v,r]==1 \/ Sigma[r,v]==1)); % objective constraint
%strengthening...
constraint forall(s in Skills, r in Resources)((Sigma[r,s]==1) -> Y[s,r]==0);
% Objective Component Constraints
constraint sum(a in Activities, r in Resources)(weights[r,a]*X[a,r]) == Z; % objective constraint
constraint sum(s in Skills, r in Resources)(learning[r,s]*Y[s,r]) == L; % objective constraint

% Objective function
solve minimize Z+L;

output [show(Z+L)++"\n"];%, show(L)];

output["Assignments:\n"];
output [ show_float(6, 2, X[a,r]) ++
if r mod R == 0 then "\n" else " " endif |
a in Activities, r in Resources];
output["Skills:\n"];
output [show_float(6, 2, Y[s,r]) ++
if r mod R == 0 then "\n" else " " endif |
s in Skills, r in Resources];

