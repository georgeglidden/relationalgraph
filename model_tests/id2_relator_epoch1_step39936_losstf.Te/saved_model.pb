ђЃ
┐Б
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02unknown8«ї

rel_dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ *"
shared_namerel_dense1/kernel
x
%rel_dense1/kernel/Read/ReadVariableOpReadVariableOprel_dense1/kernel*
_output_shapes
:	ђ *
dtype0
v
rel_dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namerel_dense1/bias
o
#rel_dense1/bias/Read/ReadVariableOpReadVariableOprel_dense1/bias*
_output_shapes
: *
dtype0
ј
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma
Є
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0
ї
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta
Ё
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
џ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean
Њ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
б
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance
Џ
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
~
rel_dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namerel_dense2/kernel
w
%rel_dense2/kernel/Read/ReadVariableOpReadVariableOprel_dense2/kernel*
_output_shapes

: *
dtype0
v
rel_dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namerel_dense2/bias
o
#rel_dense2/bias/Read/ReadVariableOpReadVariableOprel_dense2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ѕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*├
value╣BХ B»
ё
d1
bn1
lru
d2
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Ќ
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
*

0
1
2
3
4
5
 
8

0
1
2
3
4
5
6
7
Г

#layers
$layer_metrics
%layer_regularization_losses
trainable_variables
regularization_losses
&non_trainable_variables
	variables
'metrics
 
KI
VARIABLE_VALUErel_dense1/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUErel_dense1/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
Г

(layers
)layer_metrics
*layer_regularization_losses
trainable_variables
regularization_losses
+non_trainable_variables
	variables
,metrics
 
US
VARIABLE_VALUEbatch_normalization_4/gamma$bn1/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEbatch_normalization_4/beta#bn1/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE!batch_normalization_4/moving_mean*bn1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE%batch_normalization_4/moving_variance.bn1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
3
Г

-layers
.layer_metrics
/layer_regularization_losses
trainable_variables
regularization_losses
0non_trainable_variables
	variables
1metrics
 
 
 
Г

2layers
3layer_metrics
4layer_regularization_losses
trainable_variables
regularization_losses
5non_trainable_variables
	variables
6metrics
KI
VARIABLE_VALUErel_dense2/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUErel_dense2/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г

7layers
8layer_metrics
9layer_regularization_losses
trainable_variables
 regularization_losses
:non_trainable_variables
!	variables
;metrics

0
1
2
3
 
 

0
1
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
0
1
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
|
serving_default_input_1Placeholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
љ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1rel_dense1/kernelrel_dense1/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betarel_dense2/kernelrel_dense2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *0
f+R)
'__inference_signature_wrapper_579760716
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Љ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%rel_dense1/kernel/Read/ReadVariableOp#rel_dense1/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp%rel_dense2/kernel/Read/ReadVariableOp#rel_dense2/bias/Read/ReadVariableOpConst*
Tin
2
*
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_save_579761146
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerel_dense1/kernelrel_dense1/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancerel_dense2/kernelrel_dense2/bias*
Tin
2	*
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
GPU 2J 8ѓ *.
f)R'
%__inference__traced_restore_579761180ип
ъ
K
/__inference_leaky_re_lu_layer_call_fn_579761079

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5797605662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ж
Ѓ
.__inference_rel_dense1_layer_call_fn_579760987

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_rel_dense1_layer_call_and_return_conditional_losses_5797605102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
є
Ќ
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579760485

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ђ&
Ы
%__inference__traced_restore_579761180
file_prefix&
"assignvariableop_rel_dense1_kernel&
"assignvariableop_1_rel_dense1_bias2
.assignvariableop_2_batch_normalization_4_gamma1
-assignvariableop_3_batch_normalization_4_beta8
4assignvariableop_4_batch_normalization_4_moving_mean<
8assignvariableop_5_batch_normalization_4_moving_variance(
$assignvariableop_6_rel_dense2_kernel&
"assignvariableop_7_rel_dense2_bias

identity_9ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7Р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*Ь
valueСBр	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn1/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn1/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesп
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityА
AssignVariableOpAssignVariableOp"assignvariableop_rel_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Д
AssignVariableOp_1AssignVariableOp"assignvariableop_1_rel_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_4_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3▓
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_4_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╣
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_4_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5й
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_4_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Е
AssignVariableOp_6AssignVariableOp$assignvariableop_6_rel_dense2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Д
AssignVariableOp_7AssignVariableOp"assignvariableop_7_rel_dense2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpј

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8ђ

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
д
█
+__inference_relator_layer_call_fn_579760842
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_relator_layer_call_and_return_conditional_losses_5797606532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
є
Ќ
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579761043

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕњ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѕ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Ё
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Ѓ
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
БL
ѕ
F__inference_relator_layer_call_and_return_conditional_losses_579760766
input_1-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource3
/batch_normalization_4_assignmovingavg_5797607335
1batch_normalization_4_assignmovingavg_1_579760739?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identityѕб9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpб;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp»
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02"
 rel_dense1/MatMul/ReadVariableOpЋ
rel_dense1/MatMulMatMulinput_1(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/MatMulГ
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOpГ
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/BiasAddХ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesТ
"batch_normalization_4/moments/meanMeanrel_dense1/BiasAdd:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2$
"batch_normalization_4/moments/meanЙ
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

: 2,
*batch_normalization_4/moments/StopGradientч
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencerel_dense1/BiasAdd:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:          21
/batch_normalization_4/moments/SquaredDifferenceЙ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesі
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2(
&batch_normalization_4/moments/variance┬
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╩
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1с
+batch_normalization_4/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760733*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_4/AssignMovingAvg/decayп
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_4_assignmovingavg_579760733*
_output_shapes
: *
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp┤
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760733*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/subФ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760733*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/mulІ
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_4_assignmovingavg_579760733-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760733*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_4/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760739*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_4/AssignMovingAvg_1/decayя
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_4_assignmovingavg_1_579760739*
_output_shapes
: *
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpЙ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760739*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/subх
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760739*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/mulЌ
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_4_assignmovingavg_1_579760739/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760739*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/y┌
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/addЦ
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/RsqrtЯ
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpП
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mul═
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/mul_1М
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2н
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp┘
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subП
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/add_1Ъ
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu/LeakyRelu«
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp▒
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/MatMulГ
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOpГ
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/BiasAddѓ
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rel_dense2/SigmoidС
IdentityIdentityrel_dense2/Sigmoid:y:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
╩
о
F__inference_relator_layer_call_and_return_conditional_losses_579760653
x
rel_dense1_579760632
rel_dense1_579760634#
batch_normalization_4_579760637#
batch_normalization_4_579760639#
batch_normalization_4_579760641#
batch_normalization_4_579760643
rel_dense2_579760647
rel_dense2_579760649
identityѕб-batch_normalization_4/StatefulPartitionedCallб"rel_dense1/StatefulPartitionedCallб"rel_dense2/StatefulPartitionedCallб
"rel_dense1/StatefulPartitionedCallStatefulPartitionedCallxrel_dense1_579760632rel_dense1_579760634*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_rel_dense1_layer_call_and_return_conditional_losses_5797605102$
"rel_dense1/StatefulPartitionedCall╔
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall+rel_dense1/StatefulPartitionedCall:output:0batch_normalization_4_579760637batch_normalization_4_579760639batch_normalization_4_579760641batch_normalization_4_579760643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5797604852/
-batch_normalization_4/StatefulPartitionedCallљ
leaky_re_lu/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_leaky_re_lu_layer_call_and_return_conditional_losses_5797605662
leaky_re_lu/PartitionedCall┼
"rel_dense2/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0rel_dense2_579760647rel_dense2_579760649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_rel_dense2_layer_call_and_return_conditional_losses_5797605852$
"rel_dense2/StatefulPartitionedCallщ
IdentityIdentity+rel_dense2/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall#^rel_dense1/StatefulPartitionedCall#^rel_dense2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2H
"rel_dense1/StatefulPartitionedCall"rel_dense1/StatefulPartitionedCall2H
"rel_dense2/StatefulPartitionedCall"rel_dense2/StatefulPartitionedCall:K G
(
_output_shapes
:         ђ

_user_specified_namex
ђ
О
'__inference_signature_wrapper_579760716
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference__wrapped_model_5797603562
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
й
г
9__inference_batch_normalization_4_layer_call_fn_579761069

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5797604852
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ћ
Н
+__inference_relator_layer_call_fn_579760947
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_relator_layer_call_and_return_conditional_losses_5797606532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:         ђ

_user_specified_namex
д
█
+__inference_relator_layer_call_fn_579760821
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_relator_layer_call_and_return_conditional_losses_5797606532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
ЉL
ѓ
F__inference_relator_layer_call_and_return_conditional_losses_579760892
x-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource3
/batch_normalization_4_assignmovingavg_5797608595
1batch_normalization_4_assignmovingavg_1_579760865?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identityѕб9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpб;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp»
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02"
 rel_dense1/MatMul/ReadVariableOpЈ
rel_dense1/MatMulMatMulx(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/MatMulГ
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOpГ
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/BiasAddХ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesТ
"batch_normalization_4/moments/meanMeanrel_dense1/BiasAdd:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2$
"batch_normalization_4/moments/meanЙ
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

: 2,
*batch_normalization_4/moments/StopGradientч
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencerel_dense1/BiasAdd:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:          21
/batch_normalization_4/moments/SquaredDifferenceЙ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesі
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2(
&batch_normalization_4/moments/variance┬
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╩
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1с
+batch_normalization_4/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760859*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_4/AssignMovingAvg/decayп
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_4_assignmovingavg_579760859*
_output_shapes
: *
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp┤
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760859*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/subФ
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760859*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/mulІ
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_4_assignmovingavg_579760859-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_4/AssignMovingAvg/579760859*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_4/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760865*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_4/AssignMovingAvg_1/decayя
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_4_assignmovingavg_1_579760865*
_output_shapes
: *
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpЙ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760865*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/subх
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760865*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/mulЌ
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_4_assignmovingavg_1_579760865/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_4/AssignMovingAvg_1/579760865*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/y┌
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/addЦ
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/RsqrtЯ
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpП
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mul═
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/mul_1М
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2н
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp┘
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subП
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/add_1Ъ
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu/LeakyRelu«
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp▒
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/MatMulГ
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOpГ
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/BiasAddѓ
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rel_dense2/SigmoidС
IdentityIdentityrel_dense2/Sigmoid:y:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:K G
(
_output_shapes
:         ђ

_user_specified_namex
░
▒
I__inference_rel_dense2_layer_call_and_return_conditional_losses_579760585

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Н
▒
I__inference_rel_dense1_layer_call_and_return_conditional_losses_579760978

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
у
Ѓ
.__inference_rel_dense2_layer_call_fn_579761099

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_rel_dense2_layer_call_and_return_conditional_losses_5797605852
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
░
▒
I__inference_rel_dense2_layer_call_and_return_conditional_losses_579761090

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╠)
М
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579760452

inputs
assignmovingavg_579760427
assignmovingavg_1_579760433)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:          2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1А
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/579760427*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayќ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_579760427*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpк
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/579760427*
_output_shapes
: 2
AssignMovingAvg/subй
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/579760427*
_output_shapes
: 2
AssignMovingAvg/mulЄ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_579760427AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/579760427*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpД
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/579760433*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayю
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_579760433*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/579760433*
_output_shapes
: 2
AssignMovingAvg_1/subК
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/579760433*
_output_shapes
: 2
AssignMovingAvg_1/mulЊ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_579760433AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/579760433*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
├'
а
F__inference_relator_layer_call_and_return_conditional_losses_579760800
input_1-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identityѕ»
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02"
 rel_dense1/MatMul/ReadVariableOpЋ
rel_dense1/MatMulMatMulinput_1(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/MatMulГ
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOpГ
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/BiasAddн
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/yЯ
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/addЦ
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/RsqrtЯ
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpП
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mul═
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/mul_1┌
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1П
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2┌
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2█
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subП
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/add_1Ъ
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu/LeakyRelu«
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp▒
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/MatMulГ
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOpГ
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/BiasAddѓ
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rel_dense2/Sigmoidj
IdentityIdentityrel_dense2/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ:::::::::Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
Н
▒
I__inference_rel_dense1_layer_call_and_return_conditional_losses_579760510

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ћ
Н
+__inference_relator_layer_call_fn_579760968
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_relator_layer_call_and_return_conditional_losses_5797606532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:         ђ

_user_specified_namex
╣,
Й
$__inference__wrapped_model_579760356
input_15
1relator_rel_dense1_matmul_readvariableop_resource6
2relator_rel_dense1_biasadd_readvariableop_resourceC
?relator_batch_normalization_4_batchnorm_readvariableop_resourceG
Crelator_batch_normalization_4_batchnorm_mul_readvariableop_resourceE
Arelator_batch_normalization_4_batchnorm_readvariableop_1_resourceE
Arelator_batch_normalization_4_batchnorm_readvariableop_2_resource5
1relator_rel_dense2_matmul_readvariableop_resource6
2relator_rel_dense2_biasadd_readvariableop_resource
identityѕК
(relator/rel_dense1/MatMul/ReadVariableOpReadVariableOp1relator_rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02*
(relator/rel_dense1/MatMul/ReadVariableOpГ
relator/rel_dense1/MatMulMatMulinput_10relator/rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
relator/rel_dense1/MatMul┼
)relator/rel_dense1/BiasAdd/ReadVariableOpReadVariableOp2relator_rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)relator/rel_dense1/BiasAdd/ReadVariableOp═
relator/rel_dense1/BiasAddBiasAdd#relator/rel_dense1/MatMul:product:01relator/rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
relator/rel_dense1/BiasAddВ
6relator/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?relator_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6relator/batch_normalization_4/batchnorm/ReadVariableOpБ
-relator/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2/
-relator/batch_normalization_4/batchnorm/add/yђ
+relator/batch_normalization_4/batchnorm/addAddV2>relator/batch_normalization_4/batchnorm/ReadVariableOp:value:06relator/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+relator/batch_normalization_4/batchnorm/addй
-relator/batch_normalization_4/batchnorm/RsqrtRsqrt/relator/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-relator/batch_normalization_4/batchnorm/RsqrtЭ
:relator/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCrelator_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:relator/batch_normalization_4/batchnorm/mul/ReadVariableOp§
+relator/batch_normalization_4/batchnorm/mulMul1relator/batch_normalization_4/batchnorm/Rsqrt:y:0Brelator/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+relator/batch_normalization_4/batchnorm/mulь
-relator/batch_normalization_4/batchnorm/mul_1Mul#relator/rel_dense1/BiasAdd:output:0/relator/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2/
-relator/batch_normalization_4/batchnorm/mul_1Ы
8relator/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpArelator_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8relator/batch_normalization_4/batchnorm/ReadVariableOp_1§
-relator/batch_normalization_4/batchnorm/mul_2Mul@relator/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/relator/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-relator/batch_normalization_4/batchnorm/mul_2Ы
8relator/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpArelator_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8relator/batch_normalization_4/batchnorm/ReadVariableOp_2ч
+relator/batch_normalization_4/batchnorm/subSub@relator/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01relator/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+relator/batch_normalization_4/batchnorm/sub§
-relator/batch_normalization_4/batchnorm/add_1AddV21relator/batch_normalization_4/batchnorm/mul_1:z:0/relator/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2/
-relator/batch_normalization_4/batchnorm/add_1и
relator/leaky_re_lu/LeakyRelu	LeakyRelu1relator/batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:          *
alpha%џЎЎ>2
relator/leaky_re_lu/LeakyReluк
(relator/rel_dense2/MatMul/ReadVariableOpReadVariableOp1relator_rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(relator/rel_dense2/MatMul/ReadVariableOpЛ
relator/rel_dense2/MatMulMatMul+relator/leaky_re_lu/LeakyRelu:activations:00relator/rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
relator/rel_dense2/MatMul┼
)relator/rel_dense2/BiasAdd/ReadVariableOpReadVariableOp2relator_rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)relator/rel_dense2/BiasAdd/ReadVariableOp═
relator/rel_dense2/BiasAddBiasAdd#relator/rel_dense2/MatMul:product:01relator/rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
relator/rel_dense2/BiasAddџ
relator/rel_dense2/SigmoidSigmoid#relator/rel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
relator/rel_dense2/Sigmoidr
IdentityIdentityrelator/rel_dense2/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ:::::::::Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
╗
г
9__inference_batch_normalization_4_layer_call_fn_579761056

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5797604522
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
М
f
J__inference_leaky_re_lu_layer_call_and_return_conditional_losses_579760566

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:          *
alpha%џЎЎ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ѕ
▓
"__inference__traced_save_579761146
file_prefix0
,savev2_rel_dense1_kernel_read_readvariableop.
*savev2_rel_dense1_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop0
,savev2_rel_dense2_kernel_read_readvariableop.
*savev2_rel_dense2_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_56ad5e2b31db43b8baa800c548d39941/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename▄
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*Ь
valueСBр	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn1/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn1/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesџ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesт
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_rel_dense1_kernel_read_readvariableop*savev2_rel_dense1_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop,savev2_rel_dense2_kernel_read_readvariableop*savev2_rel_dense2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*P
_input_shapes?
=: :	ђ : : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: 
М
f
J__inference_leaky_re_lu_layer_call_and_return_conditional_losses_579761074

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:          *
alpha%џЎЎ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╠)
М
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579761023

inputs
assignmovingavg_579760998
assignmovingavg_1_579761004)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб%AssignMovingAvg_1/AssignSubVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesЈ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradientц
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:          2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/varianceђ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeezeѕ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1А
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/579760998*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayќ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_579760998*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpк
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/579760998*
_output_shapes
: 2
AssignMovingAvg/subй
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/579760998*
_output_shapes
: 2
AssignMovingAvg/mulЄ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_579760998AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/579760998*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpД
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/579761004*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayю
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_579761004*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/579761004*
_output_shapes
: 2
AssignMovingAvg_1/subК
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/579761004*
_output_shapes
: 2
AssignMovingAvg_1/mulЊ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_579761004AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/579761004*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yѓ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЁ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpЂ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЁ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1х
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▒'
џ
F__inference_relator_layer_call_and_return_conditional_losses_579760926
x-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identityѕ»
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype02"
 rel_dense1/MatMul/ReadVariableOpЈ
rel_dense1/MatMulMatMulx(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/MatMulГ
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOpГ
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rel_dense1/BiasAddн
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/yЯ
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/addЦ
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/RsqrtЯ
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpП
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mul═
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/mul_1┌
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1П
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2┌
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2█
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subП
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_4/batchnorm/add_1Ъ
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu/LeakyRelu«
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp▒
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/MatMulГ
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOpГ
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rel_dense2/BiasAddѓ
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rel_dense2/Sigmoidj
IdentityIdentityrel_dense2/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:         ђ:::::::::K G
(
_output_shapes
:         ђ

_user_specified_namex"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*г
serving_defaultў
<
input_11
serving_default_input_1:0         ђ<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:Фo
Н
d1
bn1
lru
d2
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
*<&call_and_return_all_conditional_losses
=__call__
>_default_save_signature"э
_tf_keras_modelП{"class_name": "Relator", "name": "relator", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Relator"}}
Щ


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"Н
_tf_keras_layer╗{"class_name": "Dense", "name": "rel_dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rel_dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
▓	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"я
_tf_keras_layer─{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
┌
trainable_variables
regularization_losses
	variables
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"╦
_tf_keras_layer▒{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
Э

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
*E&call_and_return_all_conditional_losses
F__call__"М
_tf_keras_layer╣{"class_name": "Dense", "name": "rel_dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rel_dense2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
╩

#layers
$layer_metrics
%layer_regularization_losses
trainable_variables
regularization_losses
&non_trainable_variables
	variables
'metrics
=__call__
>_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
$:"	ђ 2rel_dense1/kernel
: 2rel_dense1/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
Г

(layers
)layer_metrics
*layer_regularization_losses
trainable_variables
regularization_losses
+non_trainable_variables
	variables
,metrics
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Г

-layers
.layer_metrics
/layer_regularization_losses
trainable_variables
regularization_losses
0non_trainable_variables
	variables
1metrics
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г

2layers
3layer_metrics
4layer_regularization_losses
trainable_variables
regularization_losses
5non_trainable_variables
	variables
6metrics
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
#:! 2rel_dense2/kernel
:2rel_dense2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г

7layers
8layer_metrics
9layer_regularization_losses
trainable_variables
 regularization_losses
:non_trainable_variables
!	variables
;metrics
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
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
0
1"
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
н2Л
F__inference_relator_layer_call_and_return_conditional_losses_579760800
F__inference_relator_layer_call_and_return_conditional_losses_579760926
F__inference_relator_layer_call_and_return_conditional_losses_579760766
F__inference_relator_layer_call_and_return_conditional_losses_579760892«
Ц▓А
FullArgSpec$
argsџ
jself
jx

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
У2т
+__inference_relator_layer_call_fn_579760821
+__inference_relator_layer_call_fn_579760842
+__inference_relator_layer_call_fn_579760947
+__inference_relator_layer_call_fn_579760968«
Ц▓А
FullArgSpec$
argsџ
jself
jx

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
с2Я
$__inference__wrapped_model_579760356и
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *'б$
"і
input_1         ђ
з2­
I__inference_rel_dense1_layer_call_and_return_conditional_losses_579760978б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_rel_dense1_layer_call_fn_579760987б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Т2с
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579761023
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579761043┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
░2Г
9__inference_batch_normalization_4_layer_call_fn_579761069
9__inference_batch_normalization_4_layer_call_fn_579761056┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
З2ы
J__inference_leaky_re_lu_layer_call_and_return_conditional_losses_579761074б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘2о
/__inference_leaky_re_lu_layer_call_fn_579761079б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_rel_dense2_layer_call_and_return_conditional_losses_579761090б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_rel_dense2_layer_call_fn_579761099б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
6B4
'__inference_signature_wrapper_579760716input_1џ
$__inference__wrapped_model_579760356r
1б.
'б$
"і
input_1         ђ
ф "3ф0
.
output_1"і
output_1         ║
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579761023b3б0
)б&
 і
inputs          
p
ф "%б"
і
0          
џ ║
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_579761043b3б0
)б&
 і
inputs          
p 
ф "%б"
і
0          
џ њ
9__inference_batch_normalization_4_layer_call_fn_579761056U3б0
)б&
 і
inputs          
p
ф "і          њ
9__inference_batch_normalization_4_layer_call_fn_579761069U3б0
)б&
 і
inputs          
p 
ф "і          д
J__inference_leaky_re_lu_layer_call_and_return_conditional_losses_579761074X/б,
%б"
 і
inputs          
ф "%б"
і
0          
џ ~
/__inference_leaky_re_lu_layer_call_fn_579761079K/б,
%б"
 і
inputs          
ф "і          ф
I__inference_rel_dense1_layer_call_and_return_conditional_losses_579760978]
0б-
&б#
!і
inputs         ђ
ф "%б"
і
0          
џ ѓ
.__inference_rel_dense1_layer_call_fn_579760987P
0б-
&б#
!і
inputs         ђ
ф "і          Е
I__inference_rel_dense2_layer_call_and_return_conditional_losses_579761090\/б,
%б"
 і
inputs          
ф "%б"
і
0         
џ Ђ
.__inference_rel_dense2_layer_call_fn_579761099O/б,
%б"
 і
inputs          
ф "і         ▓
F__inference_relator_layer_call_and_return_conditional_losses_579760766h
5б2
+б(
"і
input_1         ђ
p
ф "%б"
і
0         
џ ▓
F__inference_relator_layer_call_and_return_conditional_losses_579760800h
5б2
+б(
"і
input_1         ђ
p 
ф "%б"
і
0         
џ г
F__inference_relator_layer_call_and_return_conditional_losses_579760892b
/б,
%б"
і
x         ђ
p
ф "%б"
і
0         
џ г
F__inference_relator_layer_call_and_return_conditional_losses_579760926b
/б,
%б"
і
x         ђ
p 
ф "%б"
і
0         
џ і
+__inference_relator_layer_call_fn_579760821[
5б2
+б(
"і
input_1         ђ
p
ф "і         і
+__inference_relator_layer_call_fn_579760842[
5б2
+б(
"і
input_1         ђ
p 
ф "і         ё
+__inference_relator_layer_call_fn_579760947U
/б,
%б"
і
x         ђ
p
ф "і         ё
+__inference_relator_layer_call_fn_579760968U
/б,
%б"
і
x         ђ
p 
ф "і         е
'__inference_signature_wrapper_579760716}
<б9
б 
2ф/
-
input_1"і
input_1         ђ"3ф0
.
output_1"і
output_1         