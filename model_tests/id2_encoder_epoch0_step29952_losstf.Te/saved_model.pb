��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02unknown8ɸ
�
bb_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebb_conv1/kernel
{
#bb_conv1/kernel/Read/ReadVariableOpReadVariableOpbb_conv1/kernel*&
_output_shapes
:*
dtype0
r
bb_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebb_conv1/bias
k
!bb_conv1/bias/Read/ReadVariableOpReadVariableOpbb_conv1/bias*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
bb_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebb_conv2/kernel
{
#bb_conv2/kernel/Read/ReadVariableOpReadVariableOpbb_conv2/kernel*&
_output_shapes
:*
dtype0
r
bb_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebb_conv2/bias
k
!bb_conv2/bias/Read/ReadVariableOpReadVariableOpbb_conv2/bias*
_output_shapes
:*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
�
bb_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namebb_conv3/kernel
{
#bb_conv3/kernel/Read/ReadVariableOpReadVariableOpbb_conv3/kernel*&
_output_shapes
: *
dtype0
r
bb_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebb_conv3/bias
k
!bb_conv3/bias/Read/ReadVariableOpReadVariableOpbb_conv3/bias*
_output_shapes
: *
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
}
bb_dense4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*!
shared_namebb_dense4/kernel
v
$bb_dense4/kernel/Read/ReadVariableOpReadVariableOpbb_dense4/kernel*
_output_shapes
:	�@*
dtype0
t
bb_dense4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebb_dense4/bias
m
"bb_dense4/bias/Read/ReadVariableOpReadVariableOpbb_dense4/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
�A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�@
value�@B�@ B�@
�
	conv1
	norm1
	relu1
	pool1
	conv2
	norm2
	relu2
	pool2
		conv3
	
norm3
	relu3
	pool3
flatten4

dense4
	norm4
	relu4
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
�
axis
	gamma
beta
moving_mean
 moving_variance
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
R
)trainable_variables
*regularization_losses
+	variables
,	keras_api
h

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
�
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
R
<trainable_variables
=regularization_losses
>	variables
?	keras_api
R
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
h

Dkernel
Ebias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
�
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
R
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
R
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
R
[trainable_variables
\regularization_losses
]	variables
^	keras_api
h

_kernel
`bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
�
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
R
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
v
0
1
2
3
-4
.5
46
57
D8
E9
K10
L11
_12
`13
f14
g15
 
�
0
1
2
3
4
 5
-6
.7
48
59
610
711
D12
E13
K14
L15
M16
N17
_18
`19
f20
g21
h22
i23
�

rlayers
slayer_metrics
tlayer_regularization_losses
trainable_variables
regularization_losses
unon_trainable_variables
	variables
vmetrics
 
LJ
VARIABLE_VALUEbb_conv1/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEbb_conv1/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

wlayers
xlayer_metrics
ylayer_regularization_losses
trainable_variables
regularization_losses
znon_trainable_variables
	variables
{metrics
 
US
VARIABLE_VALUEbatch_normalization/gamma&norm1/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEbatch_normalization/beta%norm1/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization/moving_mean,norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#batch_normalization/moving_variance0norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
 3
�

|layers
}layer_metrics
~layer_regularization_losses
!trainable_variables
"regularization_losses
non_trainable_variables
#	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
%trainable_variables
&regularization_losses
�non_trainable_variables
'	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
)trainable_variables
*regularization_losses
�non_trainable_variables
+	variables
�metrics
LJ
VARIABLE_VALUEbb_conv2/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEbb_conv2/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
�
�layers
�layer_metrics
 �layer_regularization_losses
/trainable_variables
0regularization_losses
�non_trainable_variables
1	variables
�metrics
 
WU
VARIABLE_VALUEbatch_normalization_1/gamma&norm2/gamma/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization_1/beta%norm2/beta/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE!batch_normalization_1/moving_mean,norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%batch_normalization_1/moving_variance0norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
62
73
�
�layers
�layer_metrics
 �layer_regularization_losses
8trainable_variables
9regularization_losses
�non_trainable_variables
:	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
<trainable_variables
=regularization_losses
�non_trainable_variables
>	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
@trainable_variables
Aregularization_losses
�non_trainable_variables
B	variables
�metrics
LJ
VARIABLE_VALUEbb_conv3/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEbb_conv3/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
�
�layers
�layer_metrics
 �layer_regularization_losses
Ftrainable_variables
Gregularization_losses
�non_trainable_variables
H	variables
�metrics
 
WU
VARIABLE_VALUEbatch_normalization_2/gamma&norm3/gamma/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization_2/beta%norm3/beta/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE!batch_normalization_2/moving_mean,norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%batch_normalization_2/moving_variance0norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
M2
N3
�
�layers
�layer_metrics
 �layer_regularization_losses
Otrainable_variables
Pregularization_losses
�non_trainable_variables
Q	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
Strainable_variables
Tregularization_losses
�non_trainable_variables
U	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
Wtrainable_variables
Xregularization_losses
�non_trainable_variables
Y	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
[trainable_variables
\regularization_losses
�non_trainable_variables
]	variables
�metrics
NL
VARIABLE_VALUEbb_dense4/kernel(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEbb_dense4/bias&dense4/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
 

_0
`1
�
�layers
�layer_metrics
 �layer_regularization_losses
atrainable_variables
bregularization_losses
�non_trainable_variables
c	variables
�metrics
 
WU
VARIABLE_VALUEbatch_normalization_3/gamma&norm4/gamma/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization_3/beta%norm4/beta/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE!batch_normalization_3/moving_mean,norm4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%batch_normalization_3/moving_variance0norm4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
h2
i3
�
�layers
�layer_metrics
 �layer_regularization_losses
jtrainable_variables
kregularization_losses
�non_trainable_variables
l	variables
�metrics
 
 
 
�
�layers
�layer_metrics
 �layer_regularization_losses
ntrainable_variables
oregularization_losses
�non_trainable_variables
p	variables
�metrics
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
8
0
 1
62
73
M4
N5
h6
i7
 
 
 
 
 
 
 
 
 

0
 1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

h0
i1
 
 
 
 
 
 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1bb_conv1/kernelbb_conv1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebb_conv2/kernelbb_conv2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebb_conv3/kernelbb_conv3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebb_dense4/kernelbb_dense4/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_182519140
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#bb_conv1/kernel/Read/ReadVariableOp!bb_conv1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#bb_conv2/kernel/Read/ReadVariableOp!bb_conv2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#bb_conv3/kernel/Read/ReadVariableOp!bb_conv3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp$bb_dense4/kernel/Read/ReadVariableOp"bb_dense4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_save_182520464
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebb_conv1/kernelbb_conv1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebb_conv2/kernelbb_conv2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebb_conv3/kernelbb_conv3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebb_dense4/kernelbb_dense4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference__traced_restore_182520546��
�
S
7__inference_average_pooling2d_1_layer_call_fn_182518154

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_1825181482
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
S
7__inference_average_pooling2d_2_layer_call_fn_182518270

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_1825182642
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
-__inference_bb_dense4_layer_call_fn_182520277

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_bb_dense4_layer_call_and_return_conditional_losses_1825187772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182518590

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������:::::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182518216

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182518685

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�)
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182520313

inputs
assignmovingavg_182520288
assignmovingavg_1_182520294)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/182520288*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182520288*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/182520288*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/182520288*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182520288AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/182520288*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/182520294*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182520294*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/182520294*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/182520294*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182520294AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/182520294*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_bb_conv2_layer_call_and_return_conditional_losses_182518537

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_bb_conv2_layer_call_and_return_conditional_losses_182519943

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_bb_conv1_layer_call_fn_182519795

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_bb_conv1_layer_call_and_return_conditional_losses_1825184242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_layer_call_fn_182519846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1825179842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520193

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_conv4_layer_call_and_return_conditional_losses_182519575
x+
'bb_conv1_conv2d_readvariableop_resource,
(bb_conv1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv2_conv2d_readvariableop_resource,
(bb_conv2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv3_conv2d_readvariableop_resource,
(bb_conv3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource,
(bb_dense4_matmul_readvariableop_resource-
)bb_dense4_biasadd_readvariableop_resource3
/batch_normalization_3_assignmovingavg_1825195495
1batch_normalization_3_assignmovingavg_1_182519555?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
bb_conv1/Conv2D/ReadVariableOpReadVariableOp'bb_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv1/Conv2D/ReadVariableOp�
bb_conv1/Conv2DConv2Dx&bb_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv1/Conv2D�
bb_conv1/BiasAdd/ReadVariableOpReadVariableOp(bb_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv1/BiasAdd/ReadVariableOp�
bb_conv1/BiasAddBiasAddbb_conv1/Conv2D:output:0'bb_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv1/BiasAdd�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3bb_conv1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1�

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

re_lu/Relu�
average_pooling2d/AvgPoolAvgPoolre_lu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
bb_conv2/Conv2D/ReadVariableOpReadVariableOp'bb_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv2/Conv2D/ReadVariableOp�
bb_conv2/Conv2DConv2D"average_pooling2d/AvgPool:output:0&bb_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv2/Conv2D�
bb_conv2/BiasAdd/ReadVariableOpReadVariableOp(bb_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv2/BiasAdd/ReadVariableOp�
bb_conv2/BiasAddBiasAddbb_conv2/Conv2D:output:0'bb_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv2/BiasAdd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3bb_conv2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1�
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
re_lu_1/Relu�
average_pooling2d_1/AvgPoolAvgPoolre_lu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
bb_conv3/Conv2D/ReadVariableOpReadVariableOp'bb_conv3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
bb_conv3/Conv2D/ReadVariableOp�
bb_conv3/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0&bb_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
bb_conv3/Conv2D�
bb_conv3/BiasAdd/ReadVariableOpReadVariableOp(bb_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
bb_conv3/BiasAdd/ReadVariableOp�
bb_conv3/BiasAddBiasAddbb_conv3/Conv2D:output:0'bb_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
bb_conv3/BiasAdd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3bb_conv3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_2/FusedBatchNormV3�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1�
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2
re_lu_2/Relu�
average_pooling2d_2/AvgPoolAvgPoolre_lu_2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshape$average_pooling2d_2/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
bb_dense4/MatMul/ReadVariableOpReadVariableOp(bb_dense4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
bb_dense4/MatMul/ReadVariableOp�
bb_dense4/MatMulMatMulflatten/Reshape:output:0'bb_dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/MatMul�
 bb_dense4/BiasAdd/ReadVariableOpReadVariableOp)bb_dense4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 bb_dense4/BiasAdd/ReadVariableOp�
bb_dense4/BiasAddBiasAddbb_dense4/MatMul:product:0(bb_dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/BiasAdd�
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indices�
"batch_normalization_3/moments/meanMeanbb_dense4/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_3/moments/mean�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradient�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencebb_dense4/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@21
/batch_normalization_3/moments/SquaredDifference�
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices�
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_3/moments/variance�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze�
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1�
+batch_normalization_3/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519549*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_3/AssignMovingAvg/decay�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_3_assignmovingavg_182519549*
_output_shapes
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519549*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/sub�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519549*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/mul�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_3_assignmovingavg_182519549-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519549*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_3/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519555*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_3/AssignMovingAvg_1/decay�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_3_assignmovingavg_1_182519555*
_output_shapes
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519555*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/sub�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519555*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/mul�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_3_assignmovingavg_1_182519555/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519555*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrt�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Mulbb_dense4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp�
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/add_1�
re_lu_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2
re_lu_3/Relu�
IdentityIdentityre_lu_3/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�
E
)__inference_re_lu_layer_call_fn_182519933

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1825185182
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�M
�	
D__inference_conv4_layer_call_and_return_conditional_losses_182518981
x
bb_conv1_182518916
bb_conv1_182518918!
batch_normalization_182518921!
batch_normalization_182518923!
batch_normalization_182518925!
batch_normalization_182518927
bb_conv2_182518932
bb_conv2_182518934#
batch_normalization_1_182518937#
batch_normalization_1_182518939#
batch_normalization_1_182518941#
batch_normalization_1_182518943
bb_conv3_182518948
bb_conv3_182518950#
batch_normalization_2_182518953#
batch_normalization_2_182518955#
batch_normalization_2_182518957#
batch_normalization_2_182518959
bb_dense4_182518965
bb_dense4_182518967#
batch_normalization_3_182518970#
batch_normalization_3_182518972#
batch_normalization_3_182518974#
batch_normalization_3_182518976
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall� bb_conv1/StatefulPartitionedCall� bb_conv2/StatefulPartitionedCall� bb_conv3/StatefulPartitionedCall�!bb_dense4/StatefulPartitionedCall�
 bb_conv1/StatefulPartitionedCallStatefulPartitionedCallxbb_conv1_182518916bb_conv1_182518918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_bb_conv1_layer_call_and_return_conditional_losses_1825184242"
 bb_conv1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)bb_conv1/StatefulPartitionedCall:output:0batch_normalization_182518921batch_normalization_182518923batch_normalization_182518925batch_normalization_182518927*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1825184772-
+batch_normalization/StatefulPartitionedCall�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1825185182
re_lu/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_1825180322#
!average_pooling2d/PartitionedCall�
 bb_conv2/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0bb_conv2_182518932bb_conv2_182518934*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_bb_conv2_layer_call_and_return_conditional_losses_1825185372"
 bb_conv2/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)bb_conv2/StatefulPartitionedCall:output:0batch_normalization_1_182518937batch_normalization_1_182518939batch_normalization_1_182518941batch_normalization_1_182518943*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825185902/
-batch_normalization_1/StatefulPartitionedCall�
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1825186312
re_lu_1/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_1825181482%
#average_pooling2d_1/PartitionedCall�
 bb_conv3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:0bb_conv3_182518948bb_conv3_182518950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_bb_conv3_layer_call_and_return_conditional_losses_1825186502"
 bb_conv3/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)bb_conv3/StatefulPartitionedCall:output:0batch_normalization_2_182518953batch_normalization_2_182518955batch_normalization_2_182518957batch_normalization_2_182518959*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1825187032/
-batch_normalization_2/StatefulPartitionedCall�
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1825187442
re_lu_2/PartitionedCall�
#average_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_1825182642%
#average_pooling2d_2/PartitionedCall�
flatten/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_1825187592
flatten/PartitionedCall�
!bb_dense4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bb_dense4_182518965bb_dense4_182518967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_bb_dense4_layer_call_and_return_conditional_losses_1825187772#
!bb_dense4/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*bb_dense4/StatefulPartitionedCall:output:0batch_normalization_3_182518970batch_normalization_3_182518972batch_normalization_3_182518974batch_normalization_3_182518976*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1825183992/
-batch_normalization_3/StatefulPartitionedCall�
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1825188332
re_lu_3/PartitionedCall�
IdentityIdentity re_lu_3/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^bb_conv1/StatefulPartitionedCall!^bb_conv2/StatefulPartitionedCall!^bb_conv3/StatefulPartitionedCall"^bb_dense4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 bb_conv1/StatefulPartitionedCall bb_conv1/StatefulPartitionedCall2D
 bb_conv2/StatefulPartitionedCall bb_conv2/StatefulPartitionedCall2D
 bb_conv3/StatefulPartitionedCall bb_conv3/StatefulPartitionedCall2F
!bb_dense4/StatefulPartitionedCall!bb_dense4/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
9__inference_batch_normalization_2_layer_call_fn_182520173

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1825182472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182518399

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@:::::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_signature_wrapper_182519140
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_1825179222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_182518744

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:��������� 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�c
�
D__inference_conv4_layer_call_and_return_conditional_losses_182519352
input_1+
'bb_conv1_conv2d_readvariableop_resource,
(bb_conv1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv2_conv2d_readvariableop_resource,
(bb_conv2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv3_conv2d_readvariableop_resource,
(bb_conv3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource,
(bb_dense4_matmul_readvariableop_resource-
)bb_dense4_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource
identity��
bb_conv1/Conv2D/ReadVariableOpReadVariableOp'bb_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv1/Conv2D/ReadVariableOp�
bb_conv1/Conv2DConv2Dinput_1&bb_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv1/Conv2D�
bb_conv1/BiasAdd/ReadVariableOpReadVariableOp(bb_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv1/BiasAdd/ReadVariableOp�
bb_conv1/BiasAddBiasAddbb_conv1/Conv2D:output:0'bb_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv1/BiasAdd�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3bb_conv1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

re_lu/Relu�
average_pooling2d/AvgPoolAvgPoolre_lu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
bb_conv2/Conv2D/ReadVariableOpReadVariableOp'bb_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv2/Conv2D/ReadVariableOp�
bb_conv2/Conv2DConv2D"average_pooling2d/AvgPool:output:0&bb_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv2/Conv2D�
bb_conv2/BiasAdd/ReadVariableOpReadVariableOp(bb_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv2/BiasAdd/ReadVariableOp�
bb_conv2/BiasAddBiasAddbb_conv2/Conv2D:output:0'bb_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv2/BiasAdd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3bb_conv2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
re_lu_1/Relu�
average_pooling2d_1/AvgPoolAvgPoolre_lu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
bb_conv3/Conv2D/ReadVariableOpReadVariableOp'bb_conv3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
bb_conv3/Conv2D/ReadVariableOp�
bb_conv3/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0&bb_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
bb_conv3/Conv2D�
bb_conv3/BiasAdd/ReadVariableOpReadVariableOp(bb_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
bb_conv3/BiasAdd/ReadVariableOp�
bb_conv3/BiasAddBiasAddbb_conv3/Conv2D:output:0'bb_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
bb_conv3/BiasAdd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3bb_conv3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3�
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2
re_lu_2/Relu�
average_pooling2d_2/AvgPoolAvgPoolre_lu_2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshape$average_pooling2d_2/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
bb_dense4/MatMul/ReadVariableOpReadVariableOp(bb_dense4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
bb_dense4/MatMul/ReadVariableOp�
bb_dense4/MatMulMatMulflatten/Reshape:output:0'bb_dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/MatMul�
 bb_dense4/BiasAdd/ReadVariableOpReadVariableOp)bb_dense4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 bb_dense4/BiasAdd/ReadVariableOp�
bb_dense4/BiasAddBiasAddbb_dense4/MatMul:product:0(bb_dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/BiasAdd�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrt�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Mulbb_dense4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/mul_1�
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1�
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2�
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2�
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/add_1�
re_lu_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2
re_lu_3/Relun
IdentityIdentityre_lu_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  :::::::::::::::::::::::::X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182519990

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_182520364

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_1_layer_call_fn_182520080

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825185902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182518572

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_182518833

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182518131

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_182518631

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_182518759

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�l
�
$__inference__wrapped_model_182517922
input_11
-conv4_bb_conv1_conv2d_readvariableop_resource2
.conv4_bb_conv1_biasadd_readvariableop_resource5
1conv4_batch_normalization_readvariableop_resource7
3conv4_batch_normalization_readvariableop_1_resourceF
Bconv4_batch_normalization_fusedbatchnormv3_readvariableop_resourceH
Dconv4_batch_normalization_fusedbatchnormv3_readvariableop_1_resource1
-conv4_bb_conv2_conv2d_readvariableop_resource2
.conv4_bb_conv2_biasadd_readvariableop_resource7
3conv4_batch_normalization_1_readvariableop_resource9
5conv4_batch_normalization_1_readvariableop_1_resourceH
Dconv4_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceJ
Fconv4_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource1
-conv4_bb_conv3_conv2d_readvariableop_resource2
.conv4_bb_conv3_biasadd_readvariableop_resource7
3conv4_batch_normalization_2_readvariableop_resource9
5conv4_batch_normalization_2_readvariableop_1_resourceH
Dconv4_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceJ
Fconv4_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource2
.conv4_bb_dense4_matmul_readvariableop_resource3
/conv4_bb_dense4_biasadd_readvariableop_resourceA
=conv4_batch_normalization_3_batchnorm_readvariableop_resourceE
Aconv4_batch_normalization_3_batchnorm_mul_readvariableop_resourceC
?conv4_batch_normalization_3_batchnorm_readvariableop_1_resourceC
?conv4_batch_normalization_3_batchnorm_readvariableop_2_resource
identity��
$conv4/bb_conv1/Conv2D/ReadVariableOpReadVariableOp-conv4_bb_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$conv4/bb_conv1/Conv2D/ReadVariableOp�
conv4/bb_conv1/Conv2DConv2Dinput_1,conv4/bb_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv4/bb_conv1/Conv2D�
%conv4/bb_conv1/BiasAdd/ReadVariableOpReadVariableOp.conv4_bb_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%conv4/bb_conv1/BiasAdd/ReadVariableOp�
conv4/bb_conv1/BiasAddBiasAddconv4/bb_conv1/Conv2D:output:0-conv4/bb_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv4/bb_conv1/BiasAdd�
(conv4/batch_normalization/ReadVariableOpReadVariableOp1conv4_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02*
(conv4/batch_normalization/ReadVariableOp�
*conv4/batch_normalization/ReadVariableOp_1ReadVariableOp3conv4_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*conv4/batch_normalization/ReadVariableOp_1�
9conv4/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBconv4_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02;
9conv4/batch_normalization/FusedBatchNormV3/ReadVariableOp�
;conv4/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDconv4_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;conv4/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
*conv4/batch_normalization/FusedBatchNormV3FusedBatchNormV3conv4/bb_conv1/BiasAdd:output:00conv4/batch_normalization/ReadVariableOp:value:02conv4/batch_normalization/ReadVariableOp_1:value:0Aconv4/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cconv4/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2,
*conv4/batch_normalization/FusedBatchNormV3�
conv4/re_lu/ReluRelu.conv4/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
conv4/re_lu/Relu�
conv4/average_pooling2d/AvgPoolAvgPoolconv4/re_lu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2!
conv4/average_pooling2d/AvgPool�
$conv4/bb_conv2/Conv2D/ReadVariableOpReadVariableOp-conv4_bb_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$conv4/bb_conv2/Conv2D/ReadVariableOp�
conv4/bb_conv2/Conv2DConv2D(conv4/average_pooling2d/AvgPool:output:0,conv4/bb_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv4/bb_conv2/Conv2D�
%conv4/bb_conv2/BiasAdd/ReadVariableOpReadVariableOp.conv4_bb_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%conv4/bb_conv2/BiasAdd/ReadVariableOp�
conv4/bb_conv2/BiasAddBiasAddconv4/bb_conv2/Conv2D:output:0-conv4/bb_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv4/bb_conv2/BiasAdd�
*conv4/batch_normalization_1/ReadVariableOpReadVariableOp3conv4_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv4/batch_normalization_1/ReadVariableOp�
,conv4/batch_normalization_1/ReadVariableOp_1ReadVariableOp5conv4_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,conv4/batch_normalization_1/ReadVariableOp_1�
;conv4/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDconv4_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02=
;conv4/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
=conv4/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFconv4_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=conv4/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
,conv4/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv4/bb_conv2/BiasAdd:output:02conv4/batch_normalization_1/ReadVariableOp:value:04conv4/batch_normalization_1/ReadVariableOp_1:value:0Cconv4/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Econv4/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2.
,conv4/batch_normalization_1/FusedBatchNormV3�
conv4/re_lu_1/ReluRelu0conv4/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
conv4/re_lu_1/Relu�
!conv4/average_pooling2d_1/AvgPoolAvgPool conv4/re_lu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2#
!conv4/average_pooling2d_1/AvgPool�
$conv4/bb_conv3/Conv2D/ReadVariableOpReadVariableOp-conv4_bb_conv3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$conv4/bb_conv3/Conv2D/ReadVariableOp�
conv4/bb_conv3/Conv2DConv2D*conv4/average_pooling2d_1/AvgPool:output:0,conv4/bb_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv4/bb_conv3/Conv2D�
%conv4/bb_conv3/BiasAdd/ReadVariableOpReadVariableOp.conv4_bb_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%conv4/bb_conv3/BiasAdd/ReadVariableOp�
conv4/bb_conv3/BiasAddBiasAddconv4/bb_conv3/Conv2D:output:0-conv4/bb_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv4/bb_conv3/BiasAdd�
*conv4/batch_normalization_2/ReadVariableOpReadVariableOp3conv4_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv4/batch_normalization_2/ReadVariableOp�
,conv4/batch_normalization_2/ReadVariableOp_1ReadVariableOp5conv4_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,conv4/batch_normalization_2/ReadVariableOp_1�
;conv4/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDconv4_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;conv4/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
=conv4/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFconv4_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=conv4/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
,conv4/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv4/bb_conv3/BiasAdd:output:02conv4/batch_normalization_2/ReadVariableOp:value:04conv4/batch_normalization_2/ReadVariableOp_1:value:0Cconv4/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Econv4/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2.
,conv4/batch_normalization_2/FusedBatchNormV3�
conv4/re_lu_2/ReluRelu0conv4/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2
conv4/re_lu_2/Relu�
!conv4/average_pooling2d_2/AvgPoolAvgPool conv4/re_lu_2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2#
!conv4/average_pooling2d_2/AvgPool{
conv4/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
conv4/flatten/Const�
conv4/flatten/ReshapeReshape*conv4/average_pooling2d_2/AvgPool:output:0conv4/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
conv4/flatten/Reshape�
%conv4/bb_dense4/MatMul/ReadVariableOpReadVariableOp.conv4_bb_dense4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02'
%conv4/bb_dense4/MatMul/ReadVariableOp�
conv4/bb_dense4/MatMulMatMulconv4/flatten/Reshape:output:0-conv4/bb_dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
conv4/bb_dense4/MatMul�
&conv4/bb_dense4/BiasAdd/ReadVariableOpReadVariableOp/conv4_bb_dense4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv4/bb_dense4/BiasAdd/ReadVariableOp�
conv4/bb_dense4/BiasAddBiasAdd conv4/bb_dense4/MatMul:product:0.conv4/bb_dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
conv4/bb_dense4/BiasAdd�
4conv4/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp=conv4_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype026
4conv4/batch_normalization_3/batchnorm/ReadVariableOp�
+conv4/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+conv4/batch_normalization_3/batchnorm/add/y�
)conv4/batch_normalization_3/batchnorm/addAddV2<conv4/batch_normalization_3/batchnorm/ReadVariableOp:value:04conv4/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2+
)conv4/batch_normalization_3/batchnorm/add�
+conv4/batch_normalization_3/batchnorm/RsqrtRsqrt-conv4/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2-
+conv4/batch_normalization_3/batchnorm/Rsqrt�
8conv4/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAconv4_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02:
8conv4/batch_normalization_3/batchnorm/mul/ReadVariableOp�
)conv4/batch_normalization_3/batchnorm/mulMul/conv4/batch_normalization_3/batchnorm/Rsqrt:y:0@conv4/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2+
)conv4/batch_normalization_3/batchnorm/mul�
+conv4/batch_normalization_3/batchnorm/mul_1Mul conv4/bb_dense4/BiasAdd:output:0-conv4/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2-
+conv4/batch_normalization_3/batchnorm/mul_1�
6conv4/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp?conv4_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6conv4/batch_normalization_3/batchnorm/ReadVariableOp_1�
+conv4/batch_normalization_3/batchnorm/mul_2Mul>conv4/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0-conv4/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2-
+conv4/batch_normalization_3/batchnorm/mul_2�
6conv4/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp?conv4_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype028
6conv4/batch_normalization_3/batchnorm/ReadVariableOp_2�
)conv4/batch_normalization_3/batchnorm/subSub>conv4/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0/conv4/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2+
)conv4/batch_normalization_3/batchnorm/sub�
+conv4/batch_normalization_3/batchnorm/add_1AddV2/conv4/batch_normalization_3/batchnorm/mul_1:z:0-conv4/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2-
+conv4/batch_normalization_3/batchnorm/add_1�
conv4/re_lu_3/ReluRelu/conv4/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2
conv4/re_lu_3/Relut
IdentityIdentity conv4/re_lu_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  :::::::::::::::::::::::::X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
`
D__inference_re_lu_layer_call_and_return_conditional_losses_182518518

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182520036

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182520333

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@:::::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_bb_conv3_layer_call_and_return_conditional_losses_182520100

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520147

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
G
+__inference_re_lu_3_layer_call_fn_182520369

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1825188332
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519897

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������:::::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_bb_conv2_layer_call_fn_182519952

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_bb_conv2_layer_call_and_return_conditional_losses_1825185372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_2_layer_call_fn_182520237

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1825187032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_1_layer_call_fn_182520016

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825181312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv4_layer_call_fn_182519776
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_1825189812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182520054

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������:::::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_3_layer_call_fn_182520359

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1825183992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_layer_call_fn_182519859

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1825180152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
l
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_182518032

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv4_layer_call_fn_182519723
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_1825189812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
G
+__inference_re_lu_2_layer_call_fn_182520247

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1825187442
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�f
�
%__inference__traced_restore_182520546
file_prefix$
 assignvariableop_bb_conv1_kernel$
 assignvariableop_1_bb_conv1_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance&
"assignvariableop_6_bb_conv2_kernel$
 assignvariableop_7_bb_conv2_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance'
#assignvariableop_12_bb_conv3_kernel%
!assignvariableop_13_bb_conv3_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance(
$assignvariableop_18_bb_dense4_kernel&
"assignvariableop_19_bb_dense4_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm3/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense4/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm4/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm4/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_bb_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_bb_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_bb_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_bb_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_bb_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_bb_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_bb_dense4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_bb_dense4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24�
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
9__inference_batch_normalization_3_layer_call_fn_182520346

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1825183662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_bb_conv3_layer_call_and_return_conditional_losses_182518650

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
n
R__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_182518264

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
H__inference_bb_dense4_layer_call_and_return_conditional_losses_182520268

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_2_layer_call_fn_182520160

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1825182162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182518703

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� :::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_layer_call_fn_182519910

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1825184592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_1_layer_call_fn_182520067

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825185722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520211

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� :::::W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
)__inference_conv4_layer_call_fn_182519458
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_1825189812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�c
�
D__inference_conv4_layer_call_and_return_conditional_losses_182519670
x+
'bb_conv1_conv2d_readvariableop_resource,
(bb_conv1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv2_conv2d_readvariableop_resource,
(bb_conv2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv3_conv2d_readvariableop_resource,
(bb_conv3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource,
(bb_dense4_matmul_readvariableop_resource-
)bb_dense4_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource
identity��
bb_conv1/Conv2D/ReadVariableOpReadVariableOp'bb_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv1/Conv2D/ReadVariableOp�
bb_conv1/Conv2DConv2Dx&bb_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv1/Conv2D�
bb_conv1/BiasAdd/ReadVariableOpReadVariableOp(bb_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv1/BiasAdd/ReadVariableOp�
bb_conv1/BiasAddBiasAddbb_conv1/Conv2D:output:0'bb_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv1/BiasAdd�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3bb_conv1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

re_lu/Relu�
average_pooling2d/AvgPoolAvgPoolre_lu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
bb_conv2/Conv2D/ReadVariableOpReadVariableOp'bb_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv2/Conv2D/ReadVariableOp�
bb_conv2/Conv2DConv2D"average_pooling2d/AvgPool:output:0&bb_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv2/Conv2D�
bb_conv2/BiasAdd/ReadVariableOpReadVariableOp(bb_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv2/BiasAdd/ReadVariableOp�
bb_conv2/BiasAddBiasAddbb_conv2/Conv2D:output:0'bb_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv2/BiasAdd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3bb_conv2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
re_lu_1/Relu�
average_pooling2d_1/AvgPoolAvgPoolre_lu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
bb_conv3/Conv2D/ReadVariableOpReadVariableOp'bb_conv3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
bb_conv3/Conv2D/ReadVariableOp�
bb_conv3/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0&bb_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
bb_conv3/Conv2D�
bb_conv3/BiasAdd/ReadVariableOpReadVariableOp(bb_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
bb_conv3/BiasAdd/ReadVariableOp�
bb_conv3/BiasAddBiasAddbb_conv3/Conv2D:output:0'bb_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
bb_conv3/BiasAdd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3bb_conv3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3�
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2
re_lu_2/Relu�
average_pooling2d_2/AvgPoolAvgPoolre_lu_2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshape$average_pooling2d_2/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
bb_dense4/MatMul/ReadVariableOpReadVariableOp(bb_dense4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
bb_dense4/MatMul/ReadVariableOp�
bb_dense4/MatMulMatMulflatten/Reshape:output:0'bb_dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/MatMul�
 bb_dense4/BiasAdd/ReadVariableOpReadVariableOp)bb_dense4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 bb_dense4/BiasAdd/ReadVariableOp�
bb_dense4/BiasAddBiasAddbb_dense4/MatMul:product:0(bb_dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/BiasAdd�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrt�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Mulbb_dense4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/mul_1�
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1�
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2�
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2�
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/add_1�
re_lu_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2
re_lu_3/Relun
IdentityIdentityre_lu_3/Relu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  :::::::::::::::::::::::::R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182518477

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������:::::W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182518247

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182518459

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_re_lu_layer_call_and_return_conditional_losses_182519928

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182518015

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������:::::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_182520085

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_bb_conv1_layer_call_and_return_conditional_losses_182519786

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  :::W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
H__inference_bb_dense4_layer_call_and_return_conditional_losses_182518777

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_conv4_layer_call_fn_182519405
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv4_layer_call_and_return_conditional_losses_1825189812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
G
+__inference_re_lu_1_layer_call_fn_182520090

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1825186312
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_182520253

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
G
+__inference_flatten_layer_call_fn_182520258

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_1825187592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_1_layer_call_fn_182520003

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825181002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182517984

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
Q
5__inference_average_pooling2d_layer_call_fn_182518038

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_1825180322
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182518100

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�)
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182518366

inputs
assignmovingavg_182518341
assignmovingavg_1_182518347)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/182518341*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182518341*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/182518341*
_output_shapes
:@2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/182518341*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182518341AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/182518341*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/182518347*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182518347*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/182518347*
_output_shapes
:@2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/182518347*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182518347AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/182518347*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182519972

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_layer_call_fn_182519923

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1825184772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_182520242

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:��������� 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
,__inference_bb_conv3_layer_call_fn_182520109

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_bb_conv3_layer_call_and_return_conditional_losses_1825186502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520129

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
n
R__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_182518148

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
D__inference_conv4_layer_call_and_return_conditional_losses_182519257
input_1+
'bb_conv1_conv2d_readvariableop_resource,
(bb_conv1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv2_conv2d_readvariableop_resource,
(bb_conv2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'bb_conv3_conv2d_readvariableop_resource,
(bb_conv3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource,
(bb_dense4_matmul_readvariableop_resource-
)bb_dense4_biasadd_readvariableop_resource3
/batch_normalization_3_assignmovingavg_1825192315
1batch_normalization_3_assignmovingavg_1_182519237?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
bb_conv1/Conv2D/ReadVariableOpReadVariableOp'bb_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv1/Conv2D/ReadVariableOp�
bb_conv1/Conv2DConv2Dinput_1&bb_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv1/Conv2D�
bb_conv1/BiasAdd/ReadVariableOpReadVariableOp(bb_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv1/BiasAdd/ReadVariableOp�
bb_conv1/BiasAddBiasAddbb_conv1/Conv2D:output:0'bb_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv1/BiasAdd�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3bb_conv1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1�

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2

re_lu/Relu�
average_pooling2d/AvgPoolAvgPoolre_lu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
bb_conv2/Conv2D/ReadVariableOpReadVariableOp'bb_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
bb_conv2/Conv2D/ReadVariableOp�
bb_conv2/Conv2DConv2D"average_pooling2d/AvgPool:output:0&bb_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
bb_conv2/Conv2D�
bb_conv2/BiasAdd/ReadVariableOpReadVariableOp(bb_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
bb_conv2/BiasAdd/ReadVariableOp�
bb_conv2/BiasAddBiasAddbb_conv2/Conv2D:output:0'bb_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
bb_conv2/BiasAdd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3bb_conv2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1�
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
re_lu_1/Relu�
average_pooling2d_1/AvgPoolAvgPoolre_lu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
bb_conv3/Conv2D/ReadVariableOpReadVariableOp'bb_conv3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
bb_conv3/Conv2D/ReadVariableOp�
bb_conv3/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0&bb_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
bb_conv3/Conv2D�
bb_conv3/BiasAdd/ReadVariableOpReadVariableOp(bb_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
bb_conv3/BiasAdd/ReadVariableOp�
bb_conv3/BiasAddBiasAddbb_conv3/Conv2D:output:0'bb_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
bb_conv3/BiasAdd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3bb_conv3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_2/FusedBatchNormV3�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1�
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� 2
re_lu_2/Relu�
average_pooling2d_2/AvgPoolAvgPoolre_lu_2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshape$average_pooling2d_2/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
bb_dense4/MatMul/ReadVariableOpReadVariableOp(bb_dense4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
bb_dense4/MatMul/ReadVariableOp�
bb_dense4/MatMulMatMulflatten/Reshape:output:0'bb_dense4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/MatMul�
 bb_dense4/BiasAdd/ReadVariableOpReadVariableOp)bb_dense4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 bb_dense4/BiasAdd/ReadVariableOp�
bb_dense4/BiasAddBiasAddbb_dense4/MatMul:product:0(bb_dense4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
bb_dense4/BiasAdd�
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indices�
"batch_normalization_3/moments/meanMeanbb_dense4/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_3/moments/mean�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradient�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencebb_dense4/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@21
/batch_normalization_3/moments/SquaredDifference�
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices�
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_3/moments/variance�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze�
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1�
+batch_normalization_3/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519231*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_3/AssignMovingAvg/decay�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_3_assignmovingavg_182519231*
_output_shapes
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519231*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/sub�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519231*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/mul�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_3_assignmovingavg_182519231-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_3/AssignMovingAvg/182519231*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_3/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519237*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_3/AssignMovingAvg_1/decay�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_3_assignmovingavg_1_182519237*
_output_shapes
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519237*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/sub�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519237*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/mul�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_3_assignmovingavg_1_182519237/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_3/AssignMovingAvg_1/182519237*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrt�
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Mulbb_dense4/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2�
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp�
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@2'
%batch_normalization_3/batchnorm/add_1�
re_lu_3/ReluRelu)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@2
re_lu_3/Relu�
IdentityIdentityre_lu_3/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*�
_input_shapes}
{:���������  ::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
G__inference_bb_conv1_layer_call_and_return_conditional_losses_182518424

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  :::W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_2_layer_call_fn_182520224

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1825186852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:��������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�8
�
"__inference__traced_save_182520464
file_prefix.
*savev2_bb_conv1_kernel_read_readvariableop,
(savev2_bb_conv1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_bb_conv2_kernel_read_readvariableop,
(savev2_bb_conv2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_bb_conv3_kernel_read_readvariableop,
(savev2_bb_conv3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop/
+savev2_bb_dense4_kernel_read_readvariableop-
)savev2_bb_dense4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_319fc8668b314a3e8e90ff705eea139a/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm3/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense4/bias/.ATTRIBUTES/VARIABLE_VALUEB&norm4/gamma/.ATTRIBUTES/VARIABLE_VALUEB%norm4/beta/.ATTRIBUTES/VARIABLE_VALUEB,norm4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB0norm4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_bb_conv1_kernel_read_readvariableop(savev2_bb_conv1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_bb_conv2_kernel_read_readvariableop(savev2_bb_conv2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_bb_conv3_kernel_read_readvariableop(savev2_bb_conv3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop+savev2_bb_dense4_kernel_read_readvariableop)savev2_bb_dense4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::: : : : : : :	�@:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: 
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������  <
output_10
StatefulPartitionedCall:0���������@tensorflow/serving/predict:�
�
	conv1
	norm1
	relu1
	pool1
	conv2
	norm2
	relu2
	pool2
		conv3
	
norm3
	relu3
	pool3
flatten4

dense4
	norm4
	relu4
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�
_tf_keras_model�{"class_name": "Conv4", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Conv4"}}
�	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "bb_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bb_conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
�	
axis
	gamma
beta
moving_mean
 moving_variance
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 8]}}
�
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "bb_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bb_conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 8]}}
�	
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 16]}}
�
<trainable_variables
=regularization_losses
>	variables
?	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

Dkernel
Ebias
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "bb_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bb_conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 16]}}
�	
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 32]}}
�
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

_kernel
`bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "bb_dense4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bb_dense4", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�	
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
0
1
2
3
-4
.5
46
57
D8
E9
K10
L11
_12
`13
f14
g15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
 5
-6
.7
48
59
610
711
D12
E13
K14
L15
M16
N17
_18
`19
f20
g21
h22
i23"
trackable_list_wrapper
�

rlayers
slayer_metrics
tlayer_regularization_losses
trainable_variables
regularization_losses
unon_trainable_variables
	variables
vmetrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
):'2bb_conv1/kernel
:2bb_conv1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

wlayers
xlayer_metrics
ylayer_regularization_losses
trainable_variables
regularization_losses
znon_trainable_variables
	variables
{metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
�

|layers
}layer_metrics
~layer_regularization_losses
!trainable_variables
"regularization_losses
non_trainable_variables
#	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
%trainable_variables
&regularization_losses
�non_trainable_variables
'	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
)trainable_variables
*regularization_losses
�non_trainable_variables
+	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2bb_conv2/kernel
:2bb_conv2/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
/trainable_variables
0regularization_losses
�non_trainable_variables
1	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
8trainable_variables
9regularization_losses
�non_trainable_variables
:	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
<trainable_variables
=regularization_losses
�non_trainable_variables
>	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
@trainable_variables
Aregularization_losses
�non_trainable_variables
B	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' 2bb_conv3/kernel
: 2bb_conv3/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
Ftrainable_variables
Gregularization_losses
�non_trainable_variables
H	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
Otrainable_variables
Pregularization_losses
�non_trainable_variables
Q	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
Strainable_variables
Tregularization_losses
�non_trainable_variables
U	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
Wtrainable_variables
Xregularization_losses
�non_trainable_variables
Y	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
[trainable_variables
\regularization_losses
�non_trainable_variables
]	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!	�@2bb_dense4/kernel
:@2bb_dense4/bias
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
atrainable_variables
bregularization_losses
�non_trainable_variables
c	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
jtrainable_variables
kregularization_losses
�non_trainable_variables
l	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layers
�layer_metrics
 �layer_regularization_losses
ntrainable_variables
oregularization_losses
�non_trainable_variables
p	variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
 1
62
73
M4
N5
h6
i7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
D__inference_conv4_layer_call_and_return_conditional_losses_182519257
D__inference_conv4_layer_call_and_return_conditional_losses_182519575
D__inference_conv4_layer_call_and_return_conditional_losses_182519352
D__inference_conv4_layer_call_and_return_conditional_losses_182519670�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv4_layer_call_fn_182519723
)__inference_conv4_layer_call_fn_182519776
)__inference_conv4_layer_call_fn_182519458
)__inference_conv4_layer_call_fn_182519405�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference__wrapped_model_182517922�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������  
�2�
G__inference_bb_conv1_layer_call_and_return_conditional_losses_182519786�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_bb_conv1_layer_call_fn_182519795�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519815
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519879
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519833
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519897�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_layer_call_fn_182519923
7__inference_batch_normalization_layer_call_fn_182519859
7__inference_batch_normalization_layer_call_fn_182519846
7__inference_batch_normalization_layer_call_fn_182519910�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_re_lu_layer_call_and_return_conditional_losses_182519928�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_re_lu_layer_call_fn_182519933�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_182518032�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
5__inference_average_pooling2d_layer_call_fn_182518038�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_bb_conv2_layer_call_and_return_conditional_losses_182519943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_bb_conv2_layer_call_fn_182519952�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182519990
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182519972
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182520054
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182520036�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
9__inference_batch_normalization_1_layer_call_fn_182520016
9__inference_batch_normalization_1_layer_call_fn_182520067
9__inference_batch_normalization_1_layer_call_fn_182520003
9__inference_batch_normalization_1_layer_call_fn_182520080�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_re_lu_1_layer_call_and_return_conditional_losses_182520085�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_re_lu_1_layer_call_fn_182520090�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_182518148�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
7__inference_average_pooling2d_1_layer_call_fn_182518154�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_bb_conv3_layer_call_and_return_conditional_losses_182520100�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_bb_conv3_layer_call_fn_182520109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520147
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520129
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520193
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520211�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
9__inference_batch_normalization_2_layer_call_fn_182520160
9__inference_batch_normalization_2_layer_call_fn_182520173
9__inference_batch_normalization_2_layer_call_fn_182520237
9__inference_batch_normalization_2_layer_call_fn_182520224�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_re_lu_2_layer_call_and_return_conditional_losses_182520242�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_re_lu_2_layer_call_fn_182520247�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_182518264�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
7__inference_average_pooling2d_2_layer_call_fn_182518270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
F__inference_flatten_layer_call_and_return_conditional_losses_182520253�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_flatten_layer_call_fn_182520258�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_bb_dense4_layer_call_and_return_conditional_losses_182520268�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_bb_dense4_layer_call_fn_182520277�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182520313
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182520333�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
9__inference_batch_normalization_3_layer_call_fn_182520346
9__inference_batch_normalization_3_layer_call_fn_182520359�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_re_lu_3_layer_call_and_return_conditional_losses_182520364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_re_lu_3_layer_call_fn_182520369�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6B4
'__inference_signature_wrapper_182519140input_1�
$__inference__wrapped_model_182517922� -.4567DEKLMN_`ifhg8�5
.�+
)�&
input_1���������  
� "3�0
.
output_1"�
output_1���������@�
R__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_182518148�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
7__inference_average_pooling2d_1_layer_call_fn_182518154�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
R__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_182518264�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
7__inference_average_pooling2d_2_layer_call_fn_182518270�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
P__inference_average_pooling2d_layer_call_and_return_conditional_losses_182518032�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
5__inference_average_pooling2d_layer_call_fn_182518038�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182519972�4567M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182519990�4567M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182520036r4567;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182520054r4567;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
9__inference_batch_normalization_1_layer_call_fn_182520003�4567M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
9__inference_batch_normalization_1_layer_call_fn_182520016�4567M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
9__inference_batch_normalization_1_layer_call_fn_182520067e4567;�8
1�.
(�%
inputs���������
p
� " �����������
9__inference_batch_normalization_1_layer_call_fn_182520080e4567;�8
1�.
(�%
inputs���������
p 
� " �����������
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520129�KLMNM�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520147�KLMNM�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520193rKLMN;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182520211rKLMN;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
9__inference_batch_normalization_2_layer_call_fn_182520160�KLMNM�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
9__inference_batch_normalization_2_layer_call_fn_182520173�KLMNM�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
9__inference_batch_normalization_2_layer_call_fn_182520224eKLMN;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
9__inference_batch_normalization_2_layer_call_fn_182520237eKLMN;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182520313bhifg3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182520333bifhg3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
9__inference_batch_normalization_3_layer_call_fn_182520346Uhifg3�0
)�&
 �
inputs���������@
p
� "����������@�
9__inference_batch_normalization_3_layer_call_fn_182520359Uifhg3�0
)�&
 �
inputs���������@
p 
� "����������@�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519815� M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519833� M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519879r ;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
R__inference_batch_normalization_layer_call_and_return_conditional_losses_182519897r ;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
7__inference_batch_normalization_layer_call_fn_182519846� M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
7__inference_batch_normalization_layer_call_fn_182519859� M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
7__inference_batch_normalization_layer_call_fn_182519910e ;�8
1�.
(�%
inputs���������
p
� " �����������
7__inference_batch_normalization_layer_call_fn_182519923e ;�8
1�.
(�%
inputs���������
p 
� " �����������
G__inference_bb_conv1_layer_call_and_return_conditional_losses_182519786l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������
� �
,__inference_bb_conv1_layer_call_fn_182519795_7�4
-�*
(�%
inputs���������  
� " �����������
G__inference_bb_conv2_layer_call_and_return_conditional_losses_182519943l-.7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
,__inference_bb_conv2_layer_call_fn_182519952_-.7�4
-�*
(�%
inputs���������
� " �����������
G__inference_bb_conv3_layer_call_and_return_conditional_losses_182520100lDE7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
,__inference_bb_conv3_layer_call_fn_182520109_DE7�4
-�*
(�%
inputs���������
� " ���������� �
H__inference_bb_dense4_layer_call_and_return_conditional_losses_182520268]_`0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
-__inference_bb_dense4_layer_call_fn_182520277P_`0�-
&�#
!�
inputs����������
� "����������@�
D__inference_conv4_layer_call_and_return_conditional_losses_182519257 -.4567DEKLMN_`hifg<�9
2�/
)�&
input_1���������  
p
� "%�"
�
0���������@
� �
D__inference_conv4_layer_call_and_return_conditional_losses_182519352 -.4567DEKLMN_`ifhg<�9
2�/
)�&
input_1���������  
p 
� "%�"
�
0���������@
� �
D__inference_conv4_layer_call_and_return_conditional_losses_182519575y -.4567DEKLMN_`hifg6�3
,�)
#� 
x���������  
p
� "%�"
�
0���������@
� �
D__inference_conv4_layer_call_and_return_conditional_losses_182519670y -.4567DEKLMN_`ifhg6�3
,�)
#� 
x���������  
p 
� "%�"
�
0���������@
� �
)__inference_conv4_layer_call_fn_182519405r -.4567DEKLMN_`ifhg<�9
2�/
)�&
input_1���������  
p
� "����������@�
)__inference_conv4_layer_call_fn_182519458r -.4567DEKLMN_`ifhg<�9
2�/
)�&
input_1���������  
p 
� "����������@�
)__inference_conv4_layer_call_fn_182519723l -.4567DEKLMN_`ifhg6�3
,�)
#� 
x���������  
p
� "����������@�
)__inference_conv4_layer_call_fn_182519776l -.4567DEKLMN_`ifhg6�3
,�)
#� 
x���������  
p 
� "����������@�
F__inference_flatten_layer_call_and_return_conditional_losses_182520253a7�4
-�*
(�%
inputs��������� 
� "&�#
�
0����������
� �
+__inference_flatten_layer_call_fn_182520258T7�4
-�*
(�%
inputs��������� 
� "������������
F__inference_re_lu_1_layer_call_and_return_conditional_losses_182520085h7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
+__inference_re_lu_1_layer_call_fn_182520090[7�4
-�*
(�%
inputs���������
� " �����������
F__inference_re_lu_2_layer_call_and_return_conditional_losses_182520242h7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
+__inference_re_lu_2_layer_call_fn_182520247[7�4
-�*
(�%
inputs��������� 
� " ���������� �
F__inference_re_lu_3_layer_call_and_return_conditional_losses_182520364X/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� z
+__inference_re_lu_3_layer_call_fn_182520369K/�,
%�"
 �
inputs���������@
� "����������@�
D__inference_re_lu_layer_call_and_return_conditional_losses_182519928h7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
)__inference_re_lu_layer_call_fn_182519933[7�4
-�*
(�%
inputs���������
� " �����������
'__inference_signature_wrapper_182519140� -.4567DEKLMN_`ifhgC�@
� 
9�6
4
input_1)�&
input_1���������  "3�0
.
output_1"�
output_1���������@