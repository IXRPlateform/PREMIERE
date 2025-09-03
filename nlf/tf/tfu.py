import numpy as np
import tensorflow as tf

_DATA_FORMAT = 'NHWC'
_DTYPE = None


def expand_dims(arr, axes):
    """Inserts new dimensions of size 1 into a tensor's shape at the given positions `axes`.
    The positions are all intepreted w.r.t. the shape of `arr` as it is *now*, therefore the order
    of `axes` doesn't matter.
    Repetition of the same axis is possible and inserts multiple new dimensions
    in that position.

     0   1   2   3   4   5  <- meaning of positive numbers in `axes`
       X   X   X   X   X    <- current elements of `arr.shape`
    -6  -5  -4  -3  -2  -1  <- meaning of negative numbers in `axes`
    """
    ndims = arr.shape.rank
    # convert negative indices to positive and sort descending
    axes = sorted([ax if ax >= 0 else ndims + ax + 1 for ax in axes], reverse=True)
    for ax in axes:
        arr = tf.expand_dims(arr, ax)
    return arr


def reduce_mean_masked(input_tensor, is_valid, axis=None, keepdims=False, try_fast=False):
    """Compute the mean of elements across dimensions of a tensor, ignoring elements if
    the corresponding element in `mask` is False.

    In general, `K = dim(mask) <= dim(input_tensor) = L`, and `mask`'s shape must match
    the first K dimensions of `tensor`'s shape. Then `input_tensor[i1,...,iK,...iL]` is
    ignored iff `mask[i1,...,iK]` is False.
    """
    if is_valid is None:
        return nan_to_num(tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims))

    if try_fast:
        return nan_to_num(tf.cond(
            tf.reduce_all(is_valid),
            lambda: tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims),
            lambda: reduce_mean_masked(input_tensor, is_valid, axis, keepdims, try_fast=False)))

    if axis is None and not keepdims:
        if isinstance(input_tensor, tf.RaggedTensor):
            if not isinstance(is_valid, tf.RaggedTensor):
                is_valid = tf.RaggedTensor.from_tensor(is_valid, ragged_rank=1)
            return nan_to_num(tf.reduce_mean(tf.boolean_mask(input_tensor.values, is_valid.values)))
        else:
            return nan_to_num(tf.reduce_mean(tf.boolean_mask(input_tensor, is_valid)))
    n_new_dims = input_tensor.shape.rank - is_valid.shape.rank
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    replaced = tf.cast(is_valid, input_tensor.dtype) * input_tensor
    sum_valid = tf.reduce_sum(replaced, axis=axis, keepdims=keepdims)
    n_valid = tf.reduce_sum(tf.cast(is_valid, input_tensor.dtype), axis=axis, keepdims=keepdims)
    return tf.math.divide_no_nan(sum_valid, n_valid)


def mean_stdev_masked(input_tensor, is_valid, items_axis, dimensions_axis, fixed_ref=None):
    if fixed_ref is not None:
        mean = fixed_ref
    else:
        mean = reduce_mean_masked(input_tensor, is_valid, axis=items_axis, keepdims=True)
    centered = input_tensor - mean

    n_new_dims = input_tensor.shape.rank - is_valid.shape.rank
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)

    n_valid = tf.reduce_sum(tf.cast(is_valid, tf.float32), axis=items_axis, keepdims=True)
    # n_valid = tf.math.count_nonzero(
    #    is_valid, axis=items_axis, keepdims=True, dtype=input_tensor.dtype)

    sum_of_squared_deviations = reduce_sum_masked(
        tf.square(centered), is_valid, axis=[items_axis, dimensions_axis], keepdims=True)

    stdev = tf.sqrt(tf.math.divide_no_nan(sum_of_squared_deviations, n_valid) + 1e-10)
    return mean, stdev


def nan_to_num(x, *, num=0):
    return tf.where(tf.math.is_nan(x), tf.cast(num, x.dtype), x)


@tf.custom_gradient
def safe_nan_to_zero(x):
    isnan = tf.math.is_nan(x)

    def grad(upstream):
        return tf.where(isnan, tf.cast(0, x.dtype), upstream)

    return tf.where(isnan, tf.cast(0, x.dtype), x), grad


def reduce_sum_masked(input_tensor, is_valid, axis=None, keepdims=False):
    """Compute the sum of elements across dimensions of a tensor, ignoring elements if
    the corresponding element in `mask` is True.

    In general, `K = dim(mask) <= dim(input_tensor) = L`, and `mask`'s shape must match
    the first K dimensions of `tensor`'s shape. Then `input_tensor[i1,...,iK,...iL]` is
    ignored iff `mask[i1,...,iK]` is True.
    """
    if axis is None and not keepdims:
        return tf.reduce_sum(tf.boolean_mask(input_tensor, is_valid))

    n_new_dims = input_tensor.shape.rank - is_valid.shape.rank
    is_valid = expand_dims(is_valid, [-1] * n_new_dims)
    replaced = tf.cast(is_valid, input_tensor.dtype) * input_tensor
    # replaced = tf.where(is_valid, input_tensor, tf.constant(0, input_tensor.dtype))
    return tf.reduce_sum(replaced, axis=axis, keepdims=keepdims)


def get_data_format():
    return _DATA_FORMAT


def set_data_format(df):
    global _DATA_FORMAT
    _DATA_FORMAT = df


def get_dtype():
    return _DTYPE


def set_dtype(dtype):
    global _DTYPE
    _DTYPE = dtype


def channel_axis():
    return _DATA_FORMAT.index('C')


def image_axes():
    return _DATA_FORMAT.index('H'), _DATA_FORMAT.index('W')


# NHWC <-> NCHW conversions
def nhwc_to_nchw(x):
    if isinstance(x, tf.Tensor):
        ndims = x.shape.rank
        if ndims == 3:
            return tf.transpose(x, [2, 0, 1])
        elif ndims == 4:
            return tf.transpose(x, [0, 3, 1, 2])
        elif ndims == 2:
            return x
        else:
            raise Exception()

    if isinstance(x, list) or isinstance(x, tuple) or x.ndim == 1:
        if len(x) == 3:
            return type(x)((x[2], x[0], x[1]))
        elif len(x) == 4:
            return type(x)((x[0], x[3], x[1], x[2]))
        elif len(x) == 2:
            return x
        raise Exception()

    if x.ndim == 3:
        return np.transpose(x, [2, 0, 1])
    elif x.ndim == 4:
        return np.transpose(x, [0, 3, 1, 2])
    elif x.ndim == 2:
        return x
    else:
        raise Exception()


def nchw_to_nhwc(x):
    if isinstance(x, tf.Tensor):
        ndims = x.shape.rank
        if ndims == 3:
            return tf.transpose(x, [1, 2, 0])
        elif ndims == 4:
            return tf.transpose(x, [0, 2, 3, 1])
        elif ndims == 2:
            return x
        else:
            raise Exception()

    if isinstance(x, list) or isinstance(x, tuple) or x.ndim == 1:
        if len(x) == 3:
            return type(x)((x[1], x[2], x[0]))
        elif len(x) == 4:
            return type(x)((x[0], x[2], x[3], x[1]))
        elif len(x) == 2:
            return x
        else:
            raise Exception()

    if x.ndim == 3:
        return np.transpose(x, [1, 2, 0])
    elif x.ndim == 4:
        return np.transpose(x, [0, 2, 3, 1])
    elif x.ndim == 2:
        return x
    else:
        raise Exception()


def convert_data_format(x, src_format=None, dst_format=None):
    src_format = src_format or _DATA_FORMAT
    dst_format = dst_format or _DATA_FORMAT

    if src_format == dst_format:
        return x
    elif src_format == 'NHWC':
        return nhwc_to_nchw(x)
    else:
        return nchw_to_nhwc(x)


def nhwc_to_std(x):
    return convert_data_format(x, src_format='NHWC')


def std_to_nhwc(x):
    return convert_data_format(x, dst_format='NHWC')


def nchw_to_std(x):
    return convert_data_format(x, src_format='NCHW')


def std_to_nchw(x):
    return convert_data_format(x, dst_format='NCHW')


def softmax(target, axis=-1):
    max_along_axis = tf.reduce_max(target, axis, keepdims=True)
    exponentiated = tf.exp(target - max_along_axis)
    denominator = tf.reduce_sum(exponentiated, axis, keepdims=True)
    return exponentiated / denominator


def soft_argmax(inp, axis):
    return decode_heatmap(softmax(inp, axis=axis), axis=axis)


def decode_heatmap(inp, axis=-1, output_coord_axis=-1):
    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else inp.shape.rank + ax for ax in axis]
    result = []
    for ax in heatmap_axes:
        other_heatmap_axes = [other_ax for other_ax in heatmap_axes if other_ax != ax]
        summed_over_other_heatmap_axes = tf.reduce_sum(inp, axis=other_heatmap_axes, keepdims=True)
        coords = tf.cast(tf.linspace(0.0, 1.0, tf.shape(inp)[ax]), inp.dtype)
        decoded = tf.tensordot(summed_over_other_heatmap_axes, coords, axes=[[ax], [0]])
        result.append(tf.squeeze(tf.expand_dims(decoded, ax), heatmap_axes))
    return tf.stack(result, axis=output_coord_axis)


def auc(x, t1, t2):
    t1 = tf.cast(t1, tf.float32)
    t2 = tf.cast(t2, tf.float32)
    return tf.nn.relu(np.float32(1) - tf.nn.relu(x - t1) / (t2 - t1))


def pck(x, t):
    return tf.reduce_mean(tf.cast(x <= t, tf.float32))


def linspace(start, stop, num, endpoint=True):
    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop, dtype=start.dtype)

    if endpoint:
        if num == 1:
            return tf.reduce_mean(tf.stack([start, stop], axis=0), axis=0, keepdims=True)
        else:
            return tf.linspace(start, stop, num)
    else:
        if num > 1:
            step = (stop - start) / tf.cast(num, start.dtype)
            return tf.linspace(start, stop - step, num)
        else:
            return tf.linspace(start, stop, num)


def make_tf_hash_table(dicti):
    keys_tensor = tf.constant(np.array(list(dicti.keys())), dtype=tf.string)
    vals_tensor_ragged = tf.ragged.constant(list(dicti.values()), dtype=tf.int32, ragged_rank=1)
    vals_tensor = vals_tensor_ragged.to_tensor(default_value=-1)
    table = tf.lookup.experimental.DenseHashTable(
        key_dtype=tf.string, value_dtype=tf.int32, empty_key='<EMPTY_SENTINEL>',
        deleted_key='<DELETE_SENTINEL>', default_value=[-1] * vals_tensor.shape[1])
    table.insert(keys_tensor, vals_tensor)
    return table


def lookup_tf_hash_table(table, key):
    key = tf.convert_to_tensor(key, dtype=tf.string)
    values = tf.RaggedTensor.from_tensor(table.lookup(key[tf.newaxis]), padding=-1)[0]
    values = tf.ensure_shape(values, [None])
    return values


def topk_indices_ragged(inp, k=1):
    row_lengths = inp.row_lengths()
    inp = inp.to_tensor(default_value=-np.inf)
    result = tf.math.top_k(inp, k=tf.minimum(k, tf.shape(inp)[1]), sorted=False).indices
    return tf.RaggedTensor.from_tensor(result, row_lengths)


def merge_last_two_axes(x):
    if not isinstance(x, tf.RaggedTensor):
        return tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], axis=0))

    values = x.values
    new_shape = tf.concat([tf.shape(values)[:-2], [-1]], axis=0)
    reshaped_values = tf.reshape(values, new_shape)
    merged_ragged = tf.RaggedTensor.from_row_splits(reshaped_values, x.row_splits)
    return merged_ragged


def unmerge_last_axis(x, n1, n2):
    if not isinstance(x, tf.RaggedTensor):
        return tf.reshape(
            x, tf.concat([tf.shape(x)[:-1], [n1, n2]], axis=0))

    values = x.values
    new_shape = tf.concat([tf.shape(values)[:-1], [n1, n2]], axis=0)
    reshaped_values = tf.reshape(values, new_shape)
    merged_ragged = tf.RaggedTensor.from_row_splits(reshaped_values, x.row_splits)
    return merged_ragged


# def euclidean_norm(x, axis=-1, keepdims=False):
#     if not isinstance(x, tf.RaggedTensor):
#         return tf.math.reduce_euclidean_norm(x, axis=axis, keepdims=keepdims)
#
#     return tf.RaggedTensor.from_row_splits(
#         tf.math.reduce_euclidean_norm(x.values, axis=axis, keepdims=keepdims), x.row_splits)


@tf.custom_gradient
def _reduce_euclidean(x):
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    def grad(upstream):
        # Divide no nan is used because the square root involved in the norm has infinite derivative
        # at zero. Instead, now the gradient will be zero at zero.
        return upstream * tf.math.divide_no_nan(x, norm)

    return norm, grad


def charbonnier(x, epsilon=1e-3, axis=-1, keepdims=False):
    if epsilon == 0:
        assert axis==-1
        return reduce_euclidean(x, keepdims) / tf.sqrt(tf.cast(tf.shape(x)[-1], x.dtype))
    else:
        return tf.sqrt(
            tf.reduce_mean(tf.square(x), axis=axis, keepdims=keepdims) + tf.square(epsilon)) - epsilon


def reduce_euclidean(x, keepdims=False):
    result = _reduce_euclidean(x)
    if not keepdims:
        result = tf.squeeze(result, axis=-1)
    return result


def to_mask_tensor(ragged, length=None):
    """Converts a tf.RaggedTensor to a dense Boolean tf.Tensor indicating whether
    each element in the dense version is valid or not."""
    row_lengths = ragged.row_lengths()
    max_length = tf.reduce_max(row_lengths) if length is None else length
    range_tensor = tf.range(max_length)
    return tf.expand_dims(range_tensor, 0) < tf.expand_dims(row_lengths, -1)


def _logcosh(x):
    return x + tf.math.softplus(-2.0 * x) - tf.cast(tf.math.log(2.0), x.dtype)


def logcosh(x, scale=1.0):
    return _logcosh(x / scale) * scale


def pad_to_size(tensor, size, axis):
    """Pads a tensor with zeros along the given axis to the given size."""
    paddings = [[0, 0]] * tensor.shape.rank
    paddings[axis] = [0, size - tf.shape(tensor)[axis]]
    return tf.pad(tensor, paddings)


def stack_coo(coo_tensors):
    indices_new = [
        tf.concat([tf.zeros_like(t.indices[:, :1]), t.indices], axis=1)
        for t in coo_tensors]
    expanded_coo_tensors = [
        tf.SparseTensor(indices=i, values=t.values, dense_shape=[1, *t.dense_shape])
        for i, t in zip(indices_new, coo_tensors)]
    return tf.sparse.concat(0, expanded_coo_tensors)


def ramp_function(x, start, length):
    if x < start:
        return tf.constant(0, tf.float32)
    elif x < start + length:
        return (tf.cast(x, tf.float32) - start) / length
    else:
        return tf.constant(1, tf.float32)
