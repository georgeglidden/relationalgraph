
æ£
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
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8£

rel_dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *"
shared_namerel_dense1/kernel
x
%rel_dense1/kernel/Read/ReadVariableOpReadVariableOprel_dense1/kernel*
_output_shapes
:	 *
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

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance

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

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ć
value¹B¶ BÆ

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

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
­

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
­

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
­

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
­

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
­

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
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1rel_dense1/kernelrel_dense1/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betarel_dense2/kernelrel_dense2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1391242867
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 *,
f'R%
#__inference__traced_save_1391243297
ķ
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
GPU 2J 8 */
f*R(
&__inference__traced_restore_1391243331©Ł

³
#__inference__traced_save_1391243297
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

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_74e47045d71a492ab40e45b4ce1f12f1/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÜ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ī
valueäBį	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn1/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn1/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_sliceså
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_rel_dense1_kernel_read_readvariableop*savev2_rel_dense1_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop,savev2_rel_dense2_kernel_read_readvariableop*savev2_rel_dense2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
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
=: :	 : : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	 : 
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
Ū)
Ö
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391242603

inputs
assignmovingavg_1391242578 
assignmovingavg_1_1391242584)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
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
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¢
AssignMovingAvg/decayConst*-
_class#
!loc:@AssignMovingAvg/1391242578*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1391242578*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpĒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*-
_class#
!loc:@AssignMovingAvg/1391242578*
_output_shapes
: 2
AssignMovingAvg/sub¾
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg/1391242578*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1391242578AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg/1391242578*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpØ
AssignMovingAvg_1/decayConst*/
_class%
#!loc:@AssignMovingAvg_1/1391242584*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1391242584*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpŃ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1391242584*
_output_shapes
: 2
AssignMovingAvg_1/subČ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1391242584*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1391242584AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*/
_class%
#!loc:@AssignMovingAvg_1/1391242584*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
&
ó
&__inference__traced_restore_1391243331
file_prefix&
"assignvariableop_rel_dense1_kernel&
"assignvariableop_1_rel_dense1_bias2
.assignvariableop_2_batch_normalization_4_gamma1
-assignvariableop_3_batch_normalization_4_beta8
4assignvariableop_4_batch_normalization_4_moving_mean<
8assignvariableop_5_batch_normalization_4_moving_variance(
$assignvariableop_6_rel_dense2_kernel&
"assignvariableop_7_rel_dense2_bias

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7ā
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ī
valueäBį	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$bn1/gamma/.ATTRIBUTES/VARIABLE_VALUEB#bn1/beta/.ATTRIBUTES/VARIABLE_VALUEB*bn1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB.bn1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesŲ
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

Identity”
AssignVariableOpAssignVariableOp"assignvariableop_rel_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_rel_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_4_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3²
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_4_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¹
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_4_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5½
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_4_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6©
AssignVariableOp_6AssignVariableOp$assignvariableop_6_rel_dense2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOp"assignvariableop_7_rel_dense2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

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

Ö
,__inference_relator_layer_call_fn_1391242993
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĄ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_relator_layer_call_and_return_conditional_losses_13912428042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ø
Ü
,__inference_relator_layer_call_fn_1391243098
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĘ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_relator_layer_call_and_return_conditional_losses_13912428042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ū)
Ö
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391243174

inputs
assignmovingavg_1391243149 
assignmovingavg_1_1391243155)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
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
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1¢
AssignMovingAvg/decayConst*-
_class#
!loc:@AssignMovingAvg/1391243149*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1391243149*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpĒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*-
_class#
!loc:@AssignMovingAvg/1391243149*
_output_shapes
: 2
AssignMovingAvg/sub¾
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg/1391243149*
_output_shapes
: 2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1391243149AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg/1391243149*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpØ
AssignMovingAvg_1/decayConst*/
_class%
#!loc:@AssignMovingAvg_1/1391243155*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1391243155*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpŃ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1391243155*
_output_shapes
: 2
AssignMovingAvg_1/subČ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1391243155*
_output_shapes
: 2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1391243155AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*/
_class%
#!loc:@AssignMovingAvg_1/1391243155*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
ß
ß
G__inference_relator_layer_call_and_return_conditional_losses_1391242804
x
rel_dense1_1391242783
rel_dense1_1391242785$
 batch_normalization_4_1391242788$
 batch_normalization_4_1391242790$
 batch_normalization_4_1391242792$
 batch_normalization_4_1391242794
rel_dense2_1391242798
rel_dense2_1391242800
identity¢-batch_normalization_4/StatefulPartitionedCall¢"rel_dense1/StatefulPartitionedCall¢"rel_dense2/StatefulPartitionedCall„
"rel_dense1/StatefulPartitionedCallStatefulPartitionedCallxrel_dense1_1391242783rel_dense1_1391242785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_rel_dense1_layer_call_and_return_conditional_losses_13912426612$
"rel_dense1/StatefulPartitionedCallĪ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall+rel_dense1/StatefulPartitionedCall:output:0 batch_normalization_4_1391242788 batch_normalization_4_1391242790 batch_normalization_4_1391242792 batch_normalization_4_1391242794*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_13912426362/
-batch_normalization_4/StatefulPartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_layer_call_and_return_conditional_losses_13912427172
leaky_re_lu/PartitionedCallČ
"rel_dense2/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0rel_dense2_1391242798rel_dense2_1391242800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_rel_dense2_layer_call_and_return_conditional_losses_13912427362$
"rel_dense2/StatefulPartitionedCallł
IdentityIdentity+rel_dense2/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall#^rel_dense1/StatefulPartitionedCall#^rel_dense2/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2H
"rel_dense1/StatefulPartitionedCall"rel_dense1/StatefulPartitionedCall2H
"rel_dense2/StatefulPartitionedCall"rel_dense2/StatefulPartitionedCall:K G
(
_output_shapes
:’’’’’’’’’

_user_specified_namex


U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391242636

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’ :::::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ø
Ü
,__inference_relator_layer_call_fn_1391243119
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĘ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_relator_layer_call_and_return_conditional_losses_13912428042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
±
²
J__inference_rel_dense2_layer_call_and_return_conditional_losses_1391243241

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
 L

G__inference_relator_layer_call_and_return_conditional_losses_1391242917
x-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource4
0batch_normalization_4_assignmovingavg_13912428846
2batch_normalization_4_assignmovingavg_1_1391242890?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identity¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpÆ
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 rel_dense1/MatMul/ReadVariableOp
rel_dense1/MatMulMatMulx(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/MatMul­
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOp­
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/BiasAdd¶
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesę
"batch_normalization_4/moments/meanMeanrel_dense1/BiasAdd:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2$
"batch_normalization_4/moments/mean¾
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

: 2,
*batch_normalization_4/moments/StopGradientū
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencerel_dense1/BiasAdd:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 21
/batch_normalization_4/moments/SquaredDifference¾
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2(
&batch_normalization_4/moments/varianceĀ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeŹ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1ä
+batch_normalization_4/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391242884*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayŁ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_4_assignmovingavg_1391242884*
_output_shapes
: *
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpµ
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391242884*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/sub¬
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391242884*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/mul
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_4_assignmovingavg_1391242884-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391242884*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpź
-batch_normalization_4/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391242890*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayß
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_4_assignmovingavg_1_1391242890*
_output_shapes
: *
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpæ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391242890*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/sub¶
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391242890*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/mul
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_4_assignmovingavg_1_1391242890/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391242890*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yŚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add„
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrtą
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mulĶ
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2Ō
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpŁ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subŻ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/add_1
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu/LeakyRelu®
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp±
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/MatMul­
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOp­
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/BiasAdd
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/Sigmoidä
IdentityIdentityrel_dense2/Sigmoid:y:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:K G
(
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ō
g
K__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1391243225

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
½
­
:__inference_batch_normalization_4_layer_call_fn_1391243207

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_13912426032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’ ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ö
²
J__inference_rel_dense1_layer_call_and_return_conditional_losses_1391243129

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±
²
J__inference_rel_dense2_layer_call_and_return_conditional_losses_1391242736

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ö
²
J__inference_rel_dense1_layer_call_and_return_conditional_losses_1391242661

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ō
g
K__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1391242717

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

Ų
(__inference_signature_wrapper_1391242867
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__wrapped_model_13912425072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
²'

G__inference_relator_layer_call_and_return_conditional_losses_1391242951
x-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identityÆ
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 rel_dense1/MatMul/ReadVariableOp
rel_dense1/MatMulMatMulx(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/MatMul­
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOp­
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/BiasAddŌ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yą
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add„
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrtą
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mulĶ
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/mul_1Ś
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ż
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2Ś
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Ū
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subŻ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/add_1
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu/LeakyRelu®
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp±
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/MatMul­
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOp­
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/BiasAdd
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/Sigmoidj
IdentityIdentityrel_dense2/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’:::::::::K G
(
_output_shapes
:’’’’’’’’’

_user_specified_namex
 
L
0__inference_leaky_re_lu_layer_call_fn_1391243230

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_layer_call_and_return_conditional_losses_13912427172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

Ö
,__inference_relator_layer_call_fn_1391242972
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĄ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_relator_layer_call_and_return_conditional_losses_13912428042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ä'
”
G__inference_relator_layer_call_and_return_conditional_losses_1391243077
input_1-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identityÆ
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 rel_dense1/MatMul/ReadVariableOp
rel_dense1/MatMulMatMulinput_1(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/MatMul­
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOp­
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/BiasAddŌ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yą
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add„
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrtą
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mulĶ
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/mul_1Ś
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Ż
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2Ś
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Ū
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subŻ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/add_1
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu/LeakyRelu®
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp±
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/MatMul­
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOp­
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/BiasAdd
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/Sigmoidj
IdentityIdentityrel_dense2/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’:::::::::Q M
(
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ė

/__inference_rel_dense1_layer_call_fn_1391243138

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_rel_dense1_layer_call_and_return_conditional_losses_13912426612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
²L

G__inference_relator_layer_call_and_return_conditional_losses_1391243043
input_1-
)rel_dense1_matmul_readvariableop_resource.
*rel_dense1_biasadd_readvariableop_resource4
0batch_normalization_4_assignmovingavg_13912430106
2batch_normalization_4_assignmovingavg_1_1391243016?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource-
)rel_dense2_matmul_readvariableop_resource.
*rel_dense2_biasadd_readvariableop_resource
identity¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpÆ
 rel_dense1/MatMul/ReadVariableOpReadVariableOp)rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 rel_dense1/MatMul/ReadVariableOp
rel_dense1/MatMulMatMulinput_1(rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/MatMul­
!rel_dense1/BiasAdd/ReadVariableOpReadVariableOp*rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!rel_dense1/BiasAdd/ReadVariableOp­
rel_dense1/BiasAddBiasAddrel_dense1/MatMul:product:0)rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
rel_dense1/BiasAdd¶
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesę
"batch_normalization_4/moments/meanMeanrel_dense1/BiasAdd:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2$
"batch_normalization_4/moments/mean¾
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

: 2,
*batch_normalization_4/moments/StopGradientū
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencerel_dense1/BiasAdd:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 21
/batch_normalization_4/moments/SquaredDifference¾
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2(
&batch_normalization_4/moments/varianceĀ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeŹ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1ä
+batch_normalization_4/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391243010*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_4/AssignMovingAvg/decayŁ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_4_assignmovingavg_1391243010*
_output_shapes
: *
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpµ
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391243010*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/sub¬
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391243010*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/mul
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_4_assignmovingavg_1391243010-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg/1391243010*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpź
-batch_normalization_4/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391243016*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_4/AssignMovingAvg_1/decayß
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_4_assignmovingavg_1_1391243016*
_output_shapes
: *
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpæ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391243016*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/sub¶
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391243016*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/mul
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_4_assignmovingavg_1_1391243016/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_4/AssignMovingAvg_1/1391243016*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yŚ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add„
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrtą
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpŻ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mulĶ
%batch_normalization_4/batchnorm/mul_1Mulrel_dense1/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/mul_1Ó
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2Ō
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpŁ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/subŻ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%batch_normalization_4/batchnorm/add_1
leaky_re_lu/LeakyRelu	LeakyRelu)batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu/LeakyRelu®
 rel_dense2/MatMul/ReadVariableOpReadVariableOp)rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 rel_dense2/MatMul/ReadVariableOp±
rel_dense2/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:0(rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/MatMul­
!rel_dense2/BiasAdd/ReadVariableOpReadVariableOp*rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rel_dense2/BiasAdd/ReadVariableOp­
rel_dense2/BiasAddBiasAddrel_dense2/MatMul:product:0)rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/BiasAdd
rel_dense2/SigmoidSigmoidrel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
rel_dense2/Sigmoidä
IdentityIdentityrel_dense2/Sigmoid:y:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:Q M
(
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1


U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391243194

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’ :::::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
ŗ,
æ
%__inference__wrapped_model_1391242507
input_15
1relator_rel_dense1_matmul_readvariableop_resource6
2relator_rel_dense1_biasadd_readvariableop_resourceC
?relator_batch_normalization_4_batchnorm_readvariableop_resourceG
Crelator_batch_normalization_4_batchnorm_mul_readvariableop_resourceE
Arelator_batch_normalization_4_batchnorm_readvariableop_1_resourceE
Arelator_batch_normalization_4_batchnorm_readvariableop_2_resource5
1relator_rel_dense2_matmul_readvariableop_resource6
2relator_rel_dense2_biasadd_readvariableop_resource
identityĒ
(relator/rel_dense1/MatMul/ReadVariableOpReadVariableOp1relator_rel_dense1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02*
(relator/rel_dense1/MatMul/ReadVariableOp­
relator/rel_dense1/MatMulMatMulinput_10relator/rel_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
relator/rel_dense1/MatMulÅ
)relator/rel_dense1/BiasAdd/ReadVariableOpReadVariableOp2relator_rel_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)relator/rel_dense1/BiasAdd/ReadVariableOpĶ
relator/rel_dense1/BiasAddBiasAdd#relator/rel_dense1/MatMul:product:01relator/rel_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
relator/rel_dense1/BiasAddģ
6relator/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?relator_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6relator/batch_normalization_4/batchnorm/ReadVariableOp£
-relator/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-relator/batch_normalization_4/batchnorm/add/y
+relator/batch_normalization_4/batchnorm/addAddV2>relator/batch_normalization_4/batchnorm/ReadVariableOp:value:06relator/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+relator/batch_normalization_4/batchnorm/add½
-relator/batch_normalization_4/batchnorm/RsqrtRsqrt/relator/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-relator/batch_normalization_4/batchnorm/Rsqrtų
:relator/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCrelator_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:relator/batch_normalization_4/batchnorm/mul/ReadVariableOpż
+relator/batch_normalization_4/batchnorm/mulMul1relator/batch_normalization_4/batchnorm/Rsqrt:y:0Brelator/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+relator/batch_normalization_4/batchnorm/mulķ
-relator/batch_normalization_4/batchnorm/mul_1Mul#relator/rel_dense1/BiasAdd:output:0/relator/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2/
-relator/batch_normalization_4/batchnorm/mul_1ņ
8relator/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpArelator_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8relator/batch_normalization_4/batchnorm/ReadVariableOp_1ż
-relator/batch_normalization_4/batchnorm/mul_2Mul@relator/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/relator/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-relator/batch_normalization_4/batchnorm/mul_2ņ
8relator/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpArelator_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8relator/batch_normalization_4/batchnorm/ReadVariableOp_2ū
+relator/batch_normalization_4/batchnorm/subSub@relator/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01relator/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+relator/batch_normalization_4/batchnorm/subż
-relator/batch_normalization_4/batchnorm/add_1AddV21relator/batch_normalization_4/batchnorm/mul_1:z:0/relator/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2/
-relator/batch_normalization_4/batchnorm/add_1·
relator/leaky_re_lu/LeakyRelu	LeakyRelu1relator/batch_normalization_4/batchnorm/add_1:z:0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
relator/leaky_re_lu/LeakyReluĘ
(relator/rel_dense2/MatMul/ReadVariableOpReadVariableOp1relator_rel_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(relator/rel_dense2/MatMul/ReadVariableOpŃ
relator/rel_dense2/MatMulMatMul+relator/leaky_re_lu/LeakyRelu:activations:00relator/rel_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
relator/rel_dense2/MatMulÅ
)relator/rel_dense2/BiasAdd/ReadVariableOpReadVariableOp2relator_rel_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)relator/rel_dense2/BiasAdd/ReadVariableOpĶ
relator/rel_dense2/BiasAddBiasAdd#relator/rel_dense2/MatMul:product:01relator/rel_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
relator/rel_dense2/BiasAdd
relator/rel_dense2/SigmoidSigmoid#relator/rel_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
relator/rel_dense2/Sigmoidr
IdentityIdentityrelator/rel_dense2/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:’’’’’’’’’:::::::::Q M
(
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
æ
­
:__inference_batch_normalization_4_layer_call_fn_1391243220

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_13912426362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’ ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
é

/__inference_rel_dense2_layer_call_fn_1391243250

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_rel_dense2_layer_call_and_return_conditional_losses_13912427362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0’’’’’’’’’<
output_10
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Óo
Õ
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
>_default_save_signature"÷
_tf_keras_modelŻ{"class_name": "Relator", "name": "relator", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Relator"}}
ś


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"Õ
_tf_keras_layer»{"class_name": "Dense", "name": "rel_dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rel_dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
²	
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
B__call__"Ž
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Ś
trainable_variables
regularization_losses
	variables
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"Ė
_tf_keras_layer±{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
ų

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
*E&call_and_return_all_conditional_losses
F__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "rel_dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rel_dense2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
Ź

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
$:"	 2rel_dense1/kernel
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
­

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
­

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
­

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
­

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
Ų2Õ
G__inference_relator_layer_call_and_return_conditional_losses_1391242951
G__inference_relator_layer_call_and_return_conditional_losses_1391242917
G__inference_relator_layer_call_and_return_conditional_losses_1391243077
G__inference_relator_layer_call_and_return_conditional_losses_1391243043®
„²”
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
,__inference_relator_layer_call_fn_1391243119
,__inference_relator_layer_call_fn_1391243098
,__inference_relator_layer_call_fn_1391242993
,__inference_relator_layer_call_fn_1391242972®
„²”
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä2į
%__inference__wrapped_model_1391242507·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *'¢$
"
input_1’’’’’’’’’
ō2ń
J__inference_rel_dense1_layer_call_and_return_conditional_losses_1391243129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ł2Ö
/__inference_rel_dense1_layer_call_fn_1391243138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
č2å
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391243194
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391243174“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
²2Æ
:__inference_batch_normalization_4_layer_call_fn_1391243220
:__inference_batch_normalization_4_layer_call_fn_1391243207“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
õ2ņ
K__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1391243225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ś2×
0__inference_leaky_re_lu_layer_call_fn_1391243230¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ō2ń
J__inference_rel_dense2_layer_call_and_return_conditional_losses_1391243241¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ł2Ö
/__inference_rel_dense2_layer_call_fn_1391243250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
7B5
(__inference_signature_wrapper_1391242867input_1
%__inference__wrapped_model_1391242507r
1¢.
'¢$
"
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’»
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391243174b3¢0
)¢&
 
inputs’’’’’’’’’ 
p
Ŗ "%¢"

0’’’’’’’’’ 
 »
U__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1391243194b3¢0
)¢&
 
inputs’’’’’’’’’ 
p 
Ŗ "%¢"

0’’’’’’’’’ 
 
:__inference_batch_normalization_4_layer_call_fn_1391243207U3¢0
)¢&
 
inputs’’’’’’’’’ 
p
Ŗ "’’’’’’’’’ 
:__inference_batch_normalization_4_layer_call_fn_1391243220U3¢0
)¢&
 
inputs’’’’’’’’’ 
p 
Ŗ "’’’’’’’’’ §
K__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1391243225X/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’ 
 
0__inference_leaky_re_lu_layer_call_fn_1391243230K/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’ «
J__inference_rel_dense1_layer_call_and_return_conditional_losses_1391243129]
0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’ 
 
/__inference_rel_dense1_layer_call_fn_1391243138P
0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ Ŗ
J__inference_rel_dense2_layer_call_and_return_conditional_losses_1391243241\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’
 
/__inference_rel_dense2_layer_call_fn_1391243250O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’­
G__inference_relator_layer_call_and_return_conditional_losses_1391242917b
/¢,
%¢"

x’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ­
G__inference_relator_layer_call_and_return_conditional_losses_1391242951b
/¢,
%¢"

x’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 ³
G__inference_relator_layer_call_and_return_conditional_losses_1391243043h
5¢2
+¢(
"
input_1’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’
 ³
G__inference_relator_layer_call_and_return_conditional_losses_1391243077h
5¢2
+¢(
"
input_1’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’
 
,__inference_relator_layer_call_fn_1391242972U
/¢,
%¢"

x’’’’’’’’’
p
Ŗ "’’’’’’’’’
,__inference_relator_layer_call_fn_1391242993U
/¢,
%¢"

x’’’’’’’’’
p 
Ŗ "’’’’’’’’’
,__inference_relator_layer_call_fn_1391243098[
5¢2
+¢(
"
input_1’’’’’’’’’
p
Ŗ "’’’’’’’’’
,__inference_relator_layer_call_fn_1391243119[
5¢2
+¢(
"
input_1’’’’’’’’’
p 
Ŗ "’’’’’’’’’©
(__inference_signature_wrapper_1391242867}
<¢9
¢ 
2Ŗ/
-
input_1"
input_1’’’’’’’’’"3Ŗ0
.
output_1"
output_1’’’’’’’’’