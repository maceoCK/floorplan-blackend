from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from PIL import Image
import networkx as nx
import base64

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Math and ASCII Art API"}), 200

@app.route('/generate_masks', methods=['POST'])
def generate_masks():
    data = request.get_json()
    rooms = data.get('Rooms', [])
    connectivity = data.get('Connectivity', [])
    boundary = data.get('Boundary', [])

    # Generate masks
    boundary_mask, self_mask, gen_mask = create_masks(rooms, connectivity, boundary)

    # Plot masks
    boundary_mask_image = plot_mask(boundary_mask)
    self_mask_image = plot_mask(self_mask)
    gen_mask_image = plot_mask(gen_mask)

    # Convert masks and images to base64 for JSON serialization
    response = {
        'boundary_mask': boundary_mask.tolist(),
        'self_mask': self_mask.tolist(),
        'gen_mask': gen_mask.tolist(),
        'boundary_mask_image': image_to_base64(boundary_mask_image),
        'self_mask_image': image_to_base64(self_mask_image),
        'gen_mask_image': image_to_base64(gen_mask_image)
    }
    return jsonify(response), 200

def create_masks(rooms, connectivity, boundary):
    # Number of corners and rooms
    num_rooms = len(rooms)
    num_corners = num_rooms * 4  # Assuming each room is a rectangle with 4 corners

    # Boundary Mask
    boundary_mask = np.ones((num_corners, num_corners))
    # Implement logic to set boundary_mask based on boundary data
    # ...

    # Self Mask
    self_mask = np.ones((num_corners, num_corners))
    for i in range(num_rooms):
        # Indices of corners belonging to the same room
        corner_indices = [i*4 + j for j in range(4)]
        for idx1 in corner_indices:
            for idx2 in corner_indices:
                self_mask[idx1, idx2] = 0  # Zero out positions belonging to the same room

    # Gen Mask
    gen_mask = np.zeros((num_corners, num_corners))
    # Implement logic to set gen_mask based on padding if necessary
    # ...

    return boundary_mask, self_mask, gen_mask

def plot_mask(mask):
    fig, ax = plt.subplots()
    im = ax.imshow(mask, cmap='viridis')
    plt.colorbar(im)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/plot_polygons', methods=['POST'])
def plot_polygons_endpoint():
    if 'npz_file' not in request.files:
        return jsonify({'error': 'No npz_file part in the request'}), 400

    npz_file = request.files['npz_file']
    data = np.load(npz_file, allow_pickle=True)

    # Plot polygons
    image = plot_polygons_from_npz(data)

    # Convert image to bytes
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

def plot_polygons_from_npz(data):
    idx = 0  # Assuming we're processing the first item
    houses = data['houses']
    corner_coords = houses[idx][:, :2]  # Get corner coordinates
    src_key_padding_mask = houses[idx][:, 91]  # Assuming padding mask is at index 91
    valid_coords = corner_coords[src_key_padding_mask == 0]

    # Get room indices and types
    room_indices = houses[idx][:, 58:90]
    room_indices_argmax = np.argmax(room_indices[src_key_padding_mask == 0], axis=1)
    room_types = houses[idx][:, 2:27]
    room_types_argmax = np.argmax(room_types[src_key_padding_mask == 0], axis=1)

    # Define colors for room types
    ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                12: '#D3A2C7', 13: '#785A67'}

    room_polys = []
    room_colors = []

    for i in np.unique(room_indices_argmax):
        coords = valid_coords[room_indices_argmax == i]
        coords = coords / 2 + 0.5  # Normalize coordinates between 0 and 1
        coords = coords * 256  # Scale to image size
        room_poly = Polygon(coords)
        room_color = ID_COLOR.get(room_types_argmax[room_indices_argmax == i][0], '#FFFFFF')
        room_polys.append(room_poly)
        room_colors.append(room_color)

    # Draw the polygons
    image = draw_polygons(room_polys, room_colors)
    return image

def draw_polygons(polygons, colors):
    fig, ax = plt.subplots()
    for poly, color in zip(polygons, colors):
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.7, fc=color, ec='black')

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis('off')

    # Save plot to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=4000)
