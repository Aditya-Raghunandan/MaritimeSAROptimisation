/**
 * seaGL.js -- the close-up sea, drawn on the graphics card (issues #79, #83).
 *
 * One full-screen pass per frame, at reduced resolution: water is smooth enough that 60 %
 * of the screen's pixels loses nothing the eye can find. Every pixel is placed in metres
 * on the water, so the sea is pinned to the map as it pans.
 *
 * WAVES AT EVERY SCALE (#83). The first version summed a dozen sine trains and faded out
 * any shorter than a few pixels, so at the zooms a search is watched at the sea went flat
 * and dark: "I would like that to look like the ocean". Seen from the air, the sea has
 * texture at every scale at once. So the surface is ten octaves of gradient noise, 2 m to
 * 1 km, each faded in only while it is at least a few pixels long (no shimmer) and not
 * wider than the screen. Each octave:
 *
 *   - is stretched along its crests and travels with the wind at the deep-water speed of a
 *     wave that long, c = sqrt(g L / 2 pi), in real time -- two trains 17 deg either side
 *     of the wind, so crests cross and the surface evolves instead of sliding;
 *   - is as steep as the wind makes it, strongest near the wind's own peak wavelength
 *     (closeUp.peakWavelengthM), with a floor so a calm sea still has swell and ripples;
 *   - is carried by the real current in search time, like everything in the water.
 *
 * Shading is what makes it read as water: slopes facing the light are lifted towards
 * teal, slopes facing away drop to deep blue, steep faces catch the sky, and where a wave
 * faces the sun there is glitter, in patches. At night it is the same sea, a little dimmer
 * and cooler, lit by a faint moon, with whitecaps that glow as this water's plankton does; the close-up key
 * says it is night. Whitecaps cover `whitecapFraction` of the sea (Monahan 1980).
 *
 * GUSTS AND CLOUD SHADOWS (#86) make the wind visible without a line on the map. Gusts are
 * patches of rougher, darker water racing downwind at about the wind's own speed, as a gust
 * looks from above. Cloud shadows drift at about cloud height's wind, faster than at the
 * surface; no data says where the clouds are, but which way and how fast they go is the real
 * wind. Shadows fall only by daylight, and there is no glitter in the shade.
 *
 * `createSea` answers null where WebGL, or high-precision floats in the fragment shader,
 * are missing; closeUpLayer.js then falls back to the 2-D texture of oceanLayer.js.
 */

const VERT = `
attribute vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 uRes;
uniform float uMpp;
uniform vec2 uCentre;
uniform vec2 uFlow;
uniform float uTime;
uniform vec2 uWindDir;
uniform float uWindSpeed;
uniform float uPeak;
uniform float uFoam;
uniform vec3 uSun;
uniform vec4 uMoon;
uniform float uNight;
uniform float uFade;
uniform sampler2D uLand;
uniform vec4 uLandBox;
uniform float uHasLand;

float hash21(vec2 p) {
  p = mod(p, 4096.0);
  p = fract(p * vec2(0.1031, 0.1030));
  p += dot(p, p.yx + 33.33);
  return fract((p.x + p.y) * p.x);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
             mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}

// A gradient per lattice point from two hashes, not an angle: no sin or cos, which were
// eight of the pass's costliest instructions per noise sample.
vec2 grad2(vec2 i) {
  return vec2(hash21(i), hash21(i + 19.19)) * 2.0 - 1.0;
}

// Gradient noise with its analytic derivative: (value, d/dx, d/dy).
vec3 gnoised(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  vec2 ga = grad2(i);
  vec2 gb = grad2(i + vec2(1.0, 0.0));
  vec2 gc = grad2(i + vec2(0.0, 1.0));
  vec2 gd = grad2(i + vec2(1.0, 1.0));
  float va = dot(ga, f);
  float vb = dot(gb, f - vec2(1.0, 0.0));
  float vc = dot(gc, f - vec2(0.0, 1.0));
  float vd = dot(gd, f - vec2(1.0, 1.0));
  float k = va - vb - vc + vd;
  return vec3(va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k,
              ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
              + du * (u.yx * k + vec2(vb, vc) - va));
}

vec2 turn(vec2 v, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

void main() {
  vec2 px = gl_FragCoord.xy - 0.5 * uRes;
  vec2 world = uCentre + px * uMpp;       // metres east and north of the scene origin
  vec2 q = world - uFlow;                 // the same point, in the moving water

  // The wind sets how steep the sea is; a floor keeps swell and ripples in a calm.
  float steep = 0.035 + 0.05 * clamp(uWindSpeed / 10.0, 0.0, 1.4);

  // Gusts: rougher patches, long along the wind, racing downwind; none in a calm.
  vec2 across = vec2(uWindDir.y, -uWindDir.x);
  vec2 gw = world - uWindDir * (uWindSpeed * 1.2 + 1.0) * uTime;
  vec2 ga = vec2(dot(gw, uWindDir) * 0.4, dot(gw, across));
  float gust = smoothstep(0.56, 0.84, 0.65 * vnoise(ga / 170.0) + 0.35 * vnoise(ga / 60.0 + 3.7))
    * clamp((uWindSpeed - 1.0) / 4.0, 0.0, 1.0);

  vec2 slope = vec2(0.0);
  for (int o = 0; o < 10; o++) {
    float lam = 2.0 * pow(2.0, float(o));             // 2 m ... 1024 m
    float lpx = lam / uMpp;
    float lod = smoothstep(2.0, 6.0, lpx) * (1.0 - smoothstep(260.0, 700.0, lpx));
    if (lod <= 0.0) continue;
    float d = log2(lam / uPeak);
    float w = 0.3 + 0.7 * exp(-d * d / 5.0);           // most roughness near the peak
    float c = sqrt(9.81 * lam / 6.2831853);            // deep-water phase speed
    for (int k = 0; k < 2; k++) {
      // Crossing trains only where the waves are big enough to see them cross.
      if (k == 1 && lpx < 14.0) break;
      float side = k == 0 ? 1.0 : -1.0;
      vec2 dir = turn(uWindDir, side * 0.3 + 0.07 * float(o));
      vec2 perp = vec2(dir.y, -dir.x);
      // Long along the crest, short across it, travelling with the wind.
      vec2 p = vec2(dot(q, dir) - c * uTime, dot(q, perp) * 0.42) / lam;
      vec3 n = gnoised(p + vec2(float(o) * 17.13, side * 31.7));
      vec2 dq = (n.y * dir + n.z * 0.42 * perp) / lam;
      // A gust roughens the short waves, not the swell.
      float rough = lam <= 32.0 ? 1.0 + 1.6 * gust : 1.0;
      slope += steep * w * rough * lam * dq * lod;
    }
  }
  // Wave groups: the sea's roughness swells and fades over a few hundred metres.
  slope *= 0.6 + 0.8 * vnoise(q / 640.0 + vec2(uTime * 0.01, 0.0));
  vec3 n = normalize(vec3(-slope, 1.0));

  float day = 1.0 - uNight;
  // Night is the same sea, a little dimmer and cooler: it still has to read as water.
  float bright = mix(0.8, 1.0, day);
  vec3 tint = mix(vec3(0.86, 0.95, 1.08), vec3(1.0), day);

  // A light that rakes across the waves so their relief shows: the sun, else the moon.
  vec3 L = uSun.z > 0.0 ? normalize(uSun) : normalize(uMoon.xyz);
  vec2 rake = length(L.xy) > 1e-3 ? normalize(L.xy) : vec2(0.6, 0.8);
  float facing = dot(-slope, rake);

  // The water: deep blue in the troughs, teal on the faces towards the light, in slow
  // patches the current carries.
  vec3 deep = vec3(0.016, 0.115, 0.200);
  vec3 lit = vec3(0.070, 0.360, 0.450);
  float m = 0.6 * vnoise(q / 460.0) + 0.4 * vnoise(q / 170.0 + 7.3);
  vec3 col = mix(deep, lit, clamp(0.42 + 2.6 * facing + 0.22 * (m - 0.5), 0.0, 1.0));

  // Steep faces catch the sky.
  vec3 sky = mix(vec3(0.20, 0.28, 0.42), vec3(0.62, 0.78, 0.92), day);
  float lowSun = day * (1.0 - smoothstep(0.05, 0.4, uSun.z));
  sky = mix(sky, vec3(0.95, 0.62, 0.42), lowSun * 0.6);
  col = mix(col, sky, clamp((1.0 - n.z) * 2.6, 0.0, 0.3));
  col *= bright * tint;
  col *= 1.0 - 0.14 * gust;

  // Cloud shadows, drifting with the wind at about cloud height: faster than at the surface.
  vec2 cw = (world - uWindDir * (uWindSpeed * 1.7 + 2.0) * uTime) / 2600.0;
  cw += 0.35 * vec2(vnoise(cw * 1.9 + 11.0), vnoise(cw * 1.9 + 23.0));
  float cloud = 0.55 * vnoise(cw) + 0.3 * vnoise(cw * 2.3 + 5.1) + 0.15 * vnoise(cw * 5.2 + 9.7);
  float shade = smoothstep(0.5, 0.66, cloud) * smoothstep(-0.02, 0.1, uSun.z);
  col *= 1.0 - 0.32 * shade;

  // Glitter where a wave faces the light, gathered in patches.
  vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
  float nh = max(dot(n, H), 0.0);
  float patchy = 0.35 + 0.65 * smoothstep(0.3, 0.75, vnoise(q / 900.0 + 3.1));
  float glint = pow(nh, 260.0) * 2.2 * patchy * (1.0 - shade);
  vec3 sparkle = uSun.z > 0.0
    ? mix(vec3(1.0, 0.97, 0.9), vec3(1.0, 0.75, 0.5), lowSun) * smoothstep(0.0, 0.1, uSun.z)
    : vec3(0.62, 0.72, 0.95) * uMoon.w;
  col += sparkle * glint;

  // Whitecaps: some cells of the water hold one, breaking and fading over 7 s, so that on
  // average uFoam of the sea is white. Stretched along the wind, trailing foam downwind.
  vec2 wr = vec2(uWindDir.y, -uWindDir.x);
  vec2 a = vec2(dot(q, uWindDir), dot(q, wr));
  float cellM = 34.0;
  vec2 cell = floor(a / cellM);
  vec2 cf = fract(a / cellM);
  float r1 = hash21(cell);
  float r2 = hash21(cell + 17.31);
  float r3 = hash21(cell + 41.73);
  float life = fract(uTime / 7.0 + r2);
  float env = smoothstep(0.0, 0.08, life) * (1.0 - smoothstep(0.25, 1.0, life));
  vec2 cc = vec2(0.5) + (vec2(r2, r3) - 0.5) * 0.36;
  vec2 dd = (cf - cc) / vec2(0.30, 0.13);
  dd.x *= dd.x > 0.0 ? 0.45 : 1.0;
  float blob = 1.0 - smoothstep(0.35, 1.0, length(dd) / (0.55 + 0.45 * env));
  float foam = step(r1, clamp(uFoam / 0.045, 0.0, 1.0)) * blob * env
    * (0.55 + 0.45 * vnoise(q / 3.5 + uTime * 0.3));
  float foamLod = smoothstep(1.6, 4.5, cellM * 0.3 / uMpp);
  vec3 foamCol = mix(vec3(0.30, 0.95, 0.80) * 0.6, vec3(0.94, 0.97, 1.0), day);
  col = mix(col, foamCol, clamp(foam * foamLod, 0.0, 1.0));
  col = mix(col, foamCol, uFoam * 0.8 * (1.0 - foamLod));

  // Thin out near land, where the photo underneath is the real thing.
  float water = 1.0;
  if (uHasLand > 0.5) {
    vec2 luv = (world - uLandBox.xy) / (uLandBox.zw - uLandBox.xy);
    if (luv.x >= 0.0 && luv.x <= 1.0 && luv.y >= 0.0 && luv.y <= 1.0) {
      water = smoothstep(0.35, 0.95, texture2D(uLand, luv).r);
    }
  }
  float alpha = uFade * water;
  gl_FragColor = vec4(col * alpha, alpha);
}`;

function compile(gl, type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(`sea shader: ${log}`);
  }
  return s;
}

/**
 * A sea renderer on `canvas`, or null where it cannot run. `draw(params)` paints one
 * frame; `setLand(mask, w, h, box)` gives it a water (255) / land (0) grid over a box of
 * world metres [x0, y0, x1, y1], rows from south to north.
 */
export function createSea(canvas) {
  let gl = null;
  try {
    gl = canvas.getContext('webgl', {
      alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
      preserveDrawingBuffer: false, powerPreference: 'low-power',
    });
  } catch {
    return null;
  }
  if (!gl) return null;
  const hp = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
  if (!hp || hp.precision < 23) return null;   // world metres need 32-bit floats

  let program;
  try {
    program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  } catch (err) {
    // Say why, once: a shader that will not compile on some driver falls back silently otherwise.
    console.warn(err.message);
    return null;
  }
  gl.useProgram(program);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const u = {};
  for (const name of ['uRes', 'uMpp', 'uCentre', 'uFlow', 'uTime', 'uWindDir', 'uWindSpeed', 'uPeak',
    'uFoam', 'uSun', 'uMoon', 'uNight', 'uFade', 'uLand', 'uLandBox', 'uHasLand']) {
    u[name] = gl.getUniformLocation(program, name);
  }

  const land = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, land);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([255]));
  gl.uniform1i(u.uLand, 0);
  let landBox = null;

  return {
    setLand(mask, w, h, box) {
      gl.bindTexture(gl.TEXTURE_2D, land);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, w, h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, mask);
      landBox = box;
    },

    clearLand() { landBox = null; },

    /** Wait for the GPU to finish what it was given: reading one pixel back forces it. */
    finish() {
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    },

    /** Whether the context was lost (a driver reset); the layer then falls back. */
    lost() { return gl.isContextLost(); },

    draw(p) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(u.uRes, canvas.width, canvas.height);
      gl.uniform1f(u.uMpp, p.mpp);
      gl.uniform2f(u.uCentre, p.centre[0], p.centre[1]);
      gl.uniform2f(u.uFlow, p.flow[0], p.flow[1]);
      gl.uniform1f(u.uTime, p.time);
      gl.uniform2f(u.uWindDir, p.windDir[0], p.windDir[1]);
      gl.uniform1f(u.uWindSpeed, p.windSpeed);
      gl.uniform1f(u.uPeak, p.peak);
      gl.uniform1f(u.uFoam, p.foam);
      gl.uniform3f(u.uSun, p.sun[0], p.sun[1], p.sun[2]);
      const moon = p.moon ?? [0.5, 0.5, 0.7, 0];
      gl.uniform4f(u.uMoon, moon[0], moon[1], moon[2], moon[3]);
      gl.uniform1f(u.uNight, p.night);
      gl.uniform1f(u.uFade, p.fade);
      gl.uniform1f(u.uHasLand, landBox ? 1 : 0);
      if (landBox) gl.uniform4f(u.uLandBox, landBox[0], landBox[1], landBox[2], landBox[3]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },

    dispose() {
      gl.deleteTexture(land);
      gl.deleteBuffer(buf);
      gl.deleteProgram(program);
      const ext = gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
    },
  };
}
