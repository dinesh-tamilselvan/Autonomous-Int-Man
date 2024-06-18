""" import tensorflow as tf

# Example input tensor shape
input_tensor = tf.constant([[10, 20, 30, 40, 50],
                            [60, 70, 80, 90, 100]])

# Determine the index for feature extraction based on the number of features
num_features = tf.shape(input_tensor)[-1]


print(f' featues: {num_features}, {(num_features.shape)}')
exit()
index = tf.cond(tf.equal(tf.math.floormod(num_features, 2), 0),
                lambda: num_features // 2,
                lambda: (num_features + 1) // 2)

# Extract the features using the determined index
extracted_features = tf.expand_dims(tf.gather(input_tensor, index, axis=-1), axis=-1)

# Print the shape of the extracted features
tf.print("Shape of extracted features:", extracted_features.shape)
tf.print("Extracted features:")
tf.print(extracted_features)
 """
 
 
""" import tensorflow as tf

# Example irregular-shaped input tensor
input_tensor = tf.constant([[10, 20, 30, 0],
                            [40, 50, 0, 0],
                            [60, 70, 80, 90]])
# Determine the number of features for each array in the batch
num_features = tf.reduce_sum(tf.cast(tf.math.not_equal(input_tensor, 0), dtype=tf.int32), axis=-1)
print(f'num: {num_features.shape},{type(num_features)}')

# Determine the index for feature extraction based on the number of features
index = tf.where(tf.equal(tf.math.floormod(num_features, 2), 0),
                 num_features // 2 - 1,  # Adjust the index
                 (num_features - 1) // 2)

# Extract the features using the determined index
batch_size = tf.shape(input_tensor)[0]
extracted_features = tf.stack([tf.expand_dims(tf.gather(input_tensor[i], index[i]), axis=0) for i in range(batch_size)])

# Print the shape of the extracted features
tf.print("Shape of input_tensor:", tf.shape(input_tensor))
tf.print("Shape of extracted features:", tf.shape(extracted_features))
tf.print("Extracted features:")
tf.print(extracted_features)
 """


""" import tensorflow as tf

@tf.function
def extract_features(input_tensor):
    # Determine the number of features for each array in the batch
    num_features = tf.reduce_sum(tf.cast(tf.math.not_equal(input_tensor, 0), dtype=tf.int32), axis=-1)

    # Determine the index for feature extraction based on the number of features
    index = tf.where(tf.equal(tf.math.floormod(num_features, 2), 0),
                     num_features // 2,
                     (num_features + 1) // 2)

    # Extract the features using the determined index
    batch_size = tf.shape(input_tensor)[0]
    indices = tf.stack([tf.range(batch_size), index - 1], axis=1)
    extracted_features = tf.expand_dims(tf.gather_nd(input_tensor, indices), axis=1)

    return extracted_features

# Example input tensor
input_data = [[10, 20, 30, 0],
              [40, 50, 0, 0],
              [60, 70, 80, 90]]

# Convert input_data to a Tensor
input_tensor = tf.constant(input_data)

# Call the @tf.function
extracted_features_result = extract_features(input_tensor)

# Print the extracted features
print("Extracted features:")
print(extracted_features_result) """



""" import tensorflow as tf


@tf.function
def extract_features(input_tensor):
    # Determine the number of features for each array in the batch
    num_features = tf.map_fn(lambda x: tf.reduce_sum(tf.cast(tf.math.not_equal(x, 0), dtype=tf.int32)), input_tensor)

    # Determine the index for feature extraction based on the number of features
    index = tf.where(tf.equal(tf.math.floormod(num_features, 2), 0),
                     num_features // 2,
                     (num_features + 1) // 2)

    # Extract the features using the determined index
    batch_size = tf.shape(input_tensor)[0]
    max_length = tf.shape(input_tensor)[1]
    batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, max_length])
    
    # Reshape index to match the shape of batch_indices
    index_reshaped = tf.reshape(index - 1, [-1, 1])
    
    # Tile index for each position in the row
    index_tiled = tf.tile(index_reshaped, [1, max_length])

    # Flatten both tensors to match the number of elements
    flat_batch_indices = tf.reshape(batch_indices, [-1])
    flat_index_tiled = tf.reshape(index_tiled, [-1])

    indices = tf.stack([flat_batch_indices, flat_index_tiled], axis=1)
    extracted_features = tf.gather_nd(tf.reshape(input_tensor, [-1, max_length]), indices)
    extracted_features = tf.expand_dims(extracted_features, axis=1)

    #    extracted_features = tf.reshape(extracted_features, [batch_size, 1])

    return extracted_features


# Example input tensor
input_data = [[10, 20, 30, 0],
              [40, 50, 0, 0],
              [60, 70, 80, 90]]

# Convert input_data to a Tensor
input_tensor = tf.constant(input_data)


# Call the @tf.function
extracted_features_result = extract_features(input_tensor)

# Print the extracted features
print("Extracted features:")
tf.print(extracted_features_result) """




# *******************************************************

""" 
import tensorflow as tf
@tf.function
def extract_features(input_tensor):
    # Determine the number of features for each array in the batch
    num_features = tf.shape(input_tensor)[1]
    
    print(f'num_feat:{num_features},{num_features.shape}')
    exit()
    # Determine the index for feature extraction (middle feature)
    index = tf.where(tf.equal(tf.math.floormod(num_features, 2), 0),
                     num_features // 2,
                     (num_features + 1) // 2)

    # Expand the index to match the batch size
    batch_size = tf.shape(input_tensor)[0]
    index = tf.expand_dims(index, axis=0)
    index = tf.tile(index, [batch_size])

    # Construct indices for feature extraction
    indices = tf.stack([tf.range(batch_size), index - 1], axis=1)

    # Extract the middle features using the determined indices
    extracted_features = tf.gather_nd(input_tensor, indices)

    return extracted_features

# Example input tensor of shape (batch, num_features, data)
input_data = [[[10, 20, 30],
               [40, 50, 60],
               [70, 80, 90]],

              [[100, 200, 300],
               [400, 500, 600],
               [700, 800, 900]]]

# Convert input_data to a Tensor
input_tensor = tf.constant(input_data)

# Call the @tf.function
extracted_features_result = extract_features(input_tensor)

# Print the extracted features
print("Extracted features:")
tf.print(extracted_features_result) 
tf.print(extracted_features_result.shape)
 """
# *******************************************************




import tensorflow as tf

#@tf.function
def extract_features(input_tensor):
    # Mask for padded elements
    mask = tf.reduce_any(tf.not_equal(input_tensor, 0), axis=-1)
    print(f'mask:{mask},{mask.shape}')
    
    # Find the indices of non-padded elements
    non_padded_indices = tf.where(mask)

    # Calculate the number of features for each array in the batch
    num_features = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-1)

    # Determine the index for feature extraction (middle feature)
    index = tf.where(tf.equal(tf.math.floormod(num_features, 2), 0),
                     num_features // 2,
                     (num_features + 1) // 2)

    # Expand the index to match the batch size
    batch_size = tf.shape(input_tensor)[0]
    index = tf.expand_dims(index, axis=1)  # Ensure shape is [batch_size, 1]
    index = tf.tile(index, [1, 1])  # Tile along the second dimension

    # Construct indices for feature extraction
    indices = tf.concat([tf.range(batch_size)[:, tf.newaxis], index - 1], axis=1)

    # Extract the middle features using the determined indices
    extracted_features = tf.gather_nd(input_tensor, indices)

    return extracted_features

# Example input tensor of shape (batch, num_features, data)
input_data = [[[10, 20, 30],
               [40, 50, 60],
               [70, 80, 90],[700, 800, 100], 
               [0, 0, 0]],

              [[100, 200, 300],
               [400, 500, 200],
               [700, 800, 100],
               [40, 50, 60],
               [70, 80, 90]]]

# Convert input_data to a Tensor
input_tensor = tf.constant(input_data)

# Call the @tf.function
extracted_features_result = extract_features(input_tensor)

# Print the extracted features
print("Extracted features (with padding only along axis 1):")
tf.print(extracted_features_result)





