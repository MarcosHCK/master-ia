#!/usr/bin/env python3
from array import array
from pathlib import Path
import base64, math, struct, zlib

ROOT = Path (__file__).resolve ().parent

SVG_PATH = ROOT / 'surface-comparison.svg'
WIDTH = 1680
HEIGHT = 660
MARGIN = 34
GAP = 26
PANEL_WIDTH = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
PANEL_TOP = 38
PANEL_HEIGHT = 574

def read_obj (path: Path):

  faces = []
  vertices = []

  for line in path.read_text (encoding = 'utf-8').splitlines ():

    if not (fields := line.split ()):
      continue

    if 'v' == fields [0]:
      vertices.append (tuple (map (float, fields [1:4])))

    elif 'f' == fields [0]:
      faces.append (tuple (int (part.split ('/') [0]) - 1 for part in fields [1:4]))

  if 64 * 64 != len (vertices) or 2 * 63 * 63 != len (faces):
    raise ValueError ('Expected a 64-by-64 OBJ height field with triangular faces')

  return vertices, faces

def lerp_color (a, b, amount):
  return tuple (round (x + (y - x) * amount) for x, y in zip (a, b))

def height_color (value, maximum):

  normalized = max (-1.0, min (1.0, value / maximum))
  neutral = (225, 229, 222)

  if 0 > normalized:

    rgb = lerp_color (neutral, (44, 104, 153), -normalized)

  else:
    rgb = lerp_color (neutral, (204, 104, 69), normalized)

  return '#%02x%02x%02x' % rgb

def project_vertices (vertices, panel_x):

  raw = []

  for x, y, z in vertices:

    raw.append ((0.8660254037844386 * (x - y), 0.5 * (x + y) - 0.70 * z, ))

  min_x = min (point [0] for point in raw )
  max_x = max (point [0] for point in raw )
  min_y = min (point [1] for point in raw )
  max_y = max (point [1] for point in raw )
  scale = min ((PANEL_WIDTH - 64) / (max_x - min_x), 382 / (max_y - min_y))

  center_x = panel_x + PANEL_WIDTH / 2
  center_y = 332

  projected = [ (center_x + (x - (min_x + max_x) / 2) * scale,
                 center_y + (y - (min_y + max_y) / 2) * scale, )
                for x, y in raw ]

  return projected

def rotate_vertices_about_z (vertices, angle_degrees = 180):

  center_x = (min (vertex [0] for vertex in vertices ) + max (vertex [0] for vertex in vertices )) / 2
  center_y = (min (vertex [1] for vertex in vertices ) + max (vertex [1] for vertex in vertices )) / 2
  angle = math.radians (angle_degrees)

  cosine, sine = math.cos (angle), math.sin (angle)
  rotated = []

  for x, y, z in vertices:

    offset_x, offset_y = x - center_x, y - center_y

    rotated.append ((center_x + cosine * offset_x - sine * offset_y,
                     center_y + sine * offset_x + cosine * offset_y,
                     z, ))

  return rotated

def triangle_color (points, maximum):
  a, b, c = (points [i] for i in range (3))

  ab = tuple (b [i] - a [i] for i in range (3))
  ac = tuple (c [i] - a [i] for i in range (3))
  normal = (ab [1] * ac [2] - ab [2] * ac [1],
            ab [2] * ac [0] - ab [0] * ac [2],
            ab [0] * ac [1] - ab [1] * ac [0], )

  length = math.sqrt (sum (component * component for component in normal)) or 1.0
  normal = tuple (component / length for component in normal)

  light = (-0.40, -0.48, 0.78)
  dot = sum (a * b for a, b in zip (normal, light)) / math.sqrt (sum (x * x for x in light))
  shade = 0.48 + 0.52 * max (0.0, dot)

  height = sum (point [2] for point in points ) / 3
  base = height_color (height, maximum)
  channels = tuple (int (base [i:i + 2], 16) for i in (1, 3, 5))

  return '#%02x%02x%02x' % tuple (round (channel * shade) for channel in channels)

def embed_png (path: Path, x, y, size):

  encoded = base64.b64encode (path.read_bytes ()).decode ('ascii')

  return (f'<image x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" '
          f'preserveAspectRatio="none" image-rendering="pixelated" '
          f'href="data:image/png;base64,{encoded}"/>')

def render_mesh (vertices, faces, panel_x):

  vertices = rotate_vertices_about_z (vertices, 180)
  projected = project_vertices (vertices, panel_x)

  panel_width = round (PANEL_WIDTH)
  raster_height = 440
  panel_y_offset = 105
  pixel_count = panel_width * raster_height

  depth_buffer = array ('f', [-math.inf]) * pixel_count
  image = bytearray (pixel_count * 4)

  minimum = min (vertex [2] for vertex in vertices )
  maximum = max (abs (vertex [2]) for vertex in vertices ) or 1.0
  plane_z = minimum - 1.5

  def screen (x, y, z):

    px = 0.8660254037844386 * (x - y)
    py = 0.5 * (x + y) - 0.70 * z

    bounds = [(0.8660254037844386 * (vx - vy), 0.5 * (vx + vy) - 0.70 * vz, )
              for vx, vy, vz in vertices ]
    min_x = min (point [0] for point in bounds )
    max_x = max (point [0] for point in bounds )
    min_y = min (point [1] for point in bounds )
    max_y = max (point [1] for point in bounds )
    scale = min ((PANEL_WIDTH - 64) / (max_x - min_x), 382 / (max_y - min_y))

    return (PANEL_WIDTH / 2 + (px - (min_x + max_x) / 2) * scale,
            332 + (py - (min_y + max_y) / 2) * scale, )

  def screen_svg (x, y, z):

    px, py = screen (x, y, z)

    return panel_x + px, py

  def png_chunk (tag, data):

    payload = tag + data

    return (struct.pack ('>I', len (data)) + payload +
            struct.pack ('>I', zlib.crc32 (payload) & 0xffffffff))

  plane = [screen (0, 0, plane_z), screen (63, 0, plane_z),
           screen (63, 63, plane_z), screen (0, 63, plane_z)]
  output = []

  plane_svg = [(panel_x + x, y) for x, y in plane]
  plane_points = ' '.join (f'{x:.2f},{y:.2f}' for x, y in plane_svg)
  output.append (f'<polygon points="{plane_points}" fill="#f1f3f5"/>')

  for offset in range (0, 64, 16):

    for start, end in (((offset, 0), (offset, 63)), ((0, offset), (63, offset))):

      x1, y1 = screen_svg (start [0], start [1], plane_z)
      x2, y2 = screen_svg (end [0], end [1], plane_z)

      output.append (f'<path d="M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}" '
                     'stroke="#cbd1d6" stroke-width="1"/>')

  # Place ground-plane axes before the opaque surface render so the mesh occludes them.
  x0, y0 = screen_svg (0, 0, plane_z)
  x1, y1 = screen_svg (63, 0, plane_z)
  x2, y2 = screen_svg (0, 63, plane_z)

  output.extend ([f'<path d="M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}" stroke="#43515d" stroke-width="2"/>',
                  f'<path d="M{x0:.2f},{y0:.2f} L{x2:.2f},{y2:.2f}" stroke="#43515d" stroke-width="2"/>',
                  f'<text x="{x1 + 5:.2f}" y="{y1 + 8:.2f}" class="axis">x</text>',
                  f'<text x="{x2 - 12:.2f}" y="{y2 + 12:.2f}" class="axis">y</text>', ])

  # Rasterize with a depth buffer so overlapping parts of the relief occlude correctly.
  for face in faces:
    points = [vertices [index] for index in face ]
    coords = [(projected [index] [0] - panel_x, projected [index] [1] - panel_y_offset, )
              for index in face ]
    (x1, y1), (x2, y2), (x3, y3) = coords
    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

    if abs (denominator) < 1e-10:
      continue

    x_min = max (0, math.floor (min (x1, x2, x3)))
    x_max = min (panel_width - 1, math.ceil (max (x1, x2, x3)))
    y_min = max (0, math.floor (min (y1, y2, y3)))
    y_max = min (raster_height - 1, math.ceil (max (y1, y2, y3)))
    camera_depth = [0.606 * point [0] + 0.606 * point [1] + 0.866 * point [2]
                    for point in points ]
    base_color = triangle_color (points, maximum)
    color = tuple (int (base_color [i:i + 2], 16) for i in (1, 3, 5))

    for py in range (y_min, y_max + 1):
      row = py * panel_width

      for px in range (x_min, x_max + 1):
        sample_x, sample_y = px + 0.5, py + 0.5
        weight_a = (((y2 - y3) * (sample_x - x3) +
                     (x3 - x2) * (sample_y - y3)) / denominator)
        weight_b = (((y3 - y1) * (sample_x - x3) +
                     (x1 - x3) * (sample_y - y3)) / denominator)
        weight_c = 1.0 - weight_a - weight_b

        if min (weight_a, weight_b, weight_c) < -1e-7:
          continue

        depth = (weight_a * camera_depth [0] + weight_b * camera_depth [1] +
                 weight_c * camera_depth [2])
        index = row + px

        if depth > depth_buffer [index]:
          depth_buffer [index] = depth
          pixel_index = index * 4
          image [pixel_index:pixel_index + 4] = bytes ((*color, 255))

  raw_png = b''.join (b'\x00' + bytes (image [row * panel_width * 4:(row + 1) * panel_width * 4])
                      for row in range (raster_height) )
  png_data = (b'\x89PNG\r\n\x1a\n' +
              png_chunk (b'IHDR', struct.pack ('>IIBBBBB', panel_width, raster_height, 8, 6, 0, 0, 0)) +
              png_chunk (b'IDAT', zlib.compress (raw_png, 9)) +
              png_chunk (b'IEND', b''))

  mesh_path = ROOT / 'surface-mesh-render.png'
  mesh_path.write_bytes (png_data)

  output.append (f'<image x="{panel_x:.1f}" y="{panel_y_offset}" width="{panel_width}" '
                 f'height="{raster_height}" href="data:image/png;base64,' +
                 base64.b64encode (png_data).decode ('ascii') + '"/>')

  return '\n'.join (output)

if '__main__' == __name__:

  required = [ROOT / 'input-image.png', ROOT / 'reference-surface.png',
              ROOT / 'prediction-prediction.obj']
  missing = [str (path) for path in required if not path.is_file ()]

  if missing:

    raise FileNotFoundError ('Missing render input(s): ' + ', '.join (missing))

  vertices, faces = read_obj (required [2])
  panel_xs = [MARGIN + i * (PANEL_WIDTH + GAP) for i in range (3)]
  image_size = 420
  image_y = 116
  svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
         f'viewBox="0 0 {WIDTH} {HEIGHT}">',
         '<defs>',
         '<linearGradient id="relief-legend" x1="0" x2="1" y1="0" y2="0">',
         '<stop offset="0%" stop-color="#2c6899"/>',
         '<stop offset="50%" stop-color="#e1e5de"/>',
         '<stop offset="100%" stop-color="#cc6845"/>',
         '</linearGradient>',
         '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#202833}'
         '.heading{font-size:24px;font-weight:700}.caption{font-size:17px;fill:#4d5966}'
         '.axis{font-size:18px;font-weight:700;fill:#43515d}</style>',
         '</defs>',
         '<rect width="100%" height="100%" fill="#ffffff"/>', ]

  titles = ['Imagen de entrada', 'Referencia del profesor', 'Superficie predicha']
  captions = ['des.png  ·  64 × 64 píxeles',
              'env.png  ·  salida de referencia',
              'Vista isométrica  ·  altura × 60  ·  giro Z 180°', ]

  for index, panel_x in enumerate (panel_xs):

    svg.extend ([f'<rect x="{panel_x:.1f}" y="{PANEL_TOP}" width="{PANEL_WIDTH:.1f}" '
                 f'height="{PANEL_HEIGHT}" rx="18" fill="#fbfcfd" stroke="#d9dfe5" stroke-width="2"/>',
                 f'<text x="{panel_x + PANEL_WIDTH / 2:.1f}" y="82" text-anchor="middle" '
                 f'class="heading">{titles [index]}</text>', ])

    if index < 2:

      png_path = required [index]
      image_x = panel_x + (PANEL_WIDTH - image_size) / 2
      svg.append (f'<rect x="{image_x - 2:.1f}" y="{image_y - 2}" width="{image_size + 4}" '
                  f'height="{image_size + 4}" rx="8" fill="#ffffff" stroke="#d9dfe5" stroke-width="2"/>')
      svg.append (embed_png (png_path, image_x, image_y, image_size))

    else:

      svg.append (render_mesh (vertices, faces, panel_x))

      legend_x = panel_x + 92
      legend_y = 548
      legend_width = PANEL_WIDTH - 184
      svg.extend ([f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="12" '
                   'rx="6" fill="url(#relief-legend)"/>',
                   f'<text x="{legend_x}" y="{legend_y + 27}" class="caption">Depresión</text>',
                   f'<text x="{legend_x + legend_width / 2}" y="{legend_y + 27}" '
                   'text-anchor="middle" class="caption">0</text>',
                   f'<text x="{legend_x + legend_width}" y="{legend_y + 27}" '
                   'text-anchor="end" class="caption">Elevación</text>', ])

    svg.append (f'<text x="{panel_x + PANEL_WIDTH / 2:.1f}" y="605" text-anchor="middle" '
                f'class="caption">{captions [index]}</text>')

  svg.extend (['<text x="840" y="644" text-anchor="middle" class="caption">'
               'Factor vertical elegido por inspección visual; no representa una escala física.</text>',
               '</svg>', ])

  SVG_PATH.write_text ('\n'.join (svg) + '\n', encoding = 'utf-8')
  print (f'Wrote {SVG_PATH}')