/**
 * seaGL.js -- the close-up sea, drawn on the graphics card (issue #79).
 *
 * One full-screen pass per frame, at reduced resolution: water is smooth, so 60 % of the
 * screen's pixels loses nothing the eye can find, and the pass costs a fraction of one
 * frame on an integrated GPU. Every pixel is placed in metres on the water, so the sea
 * is pinned to the map as it pans, and what it draws is set by closeUp.js:
 *
 *   waves        the trains of `waveComponents`, faded out below a few pixels long so
 *                the sea never shimmers with detail finer than the screen can show;
 *   the water    slow patches carried by the real current, so the water is seen to move;
 *   whitecaps    covering `whitecapFraction` of the sea, breaking and fading over seconds,
 *                drawn along the wind; at night they glow faintly green, as these waters'
 *                plankton do;
 *   light        from the real sun: glitter where a wave faces it, warm at dusk; at night
 *                a moonlit blue with a faint glitter, not black, so the sea still reads;
 *   land         the sea thins out within a current-model cell of the coast, where the
 *                satellite photo underneath is real.
 *
 * `createSea` answers null where WebGL, or high-precision floats in the fragment shader,
 * are missing; closeUpLayer.js then falls back to the 2-D texture of oceanLayer.js.
 */

const WAVES = 12;

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
uniform vec4 uWaves[${WAVES}];
uniform vec2 uWindDir;
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

void main() {
  vec2 px = gl_FragCoord.xy - 0.5 * uRes;
  vec2 world = uCentre + px * uMpp;       // metres east and north of the scene origin
  vec2 q = world - uFlow;                 // the same point, in the moving water

  // The wave trains: height and slope, each faded out below a few pixels long.
  vec2 grad = vec2(0.0);
  float h = 0.0;
  float hMax = 1e-4;
  for (int i = 0; i < ${WAVES}; i++) {
    vec4 w = uWaves[i];
    float k = length(w.xy);
    float lod = smoothstep(2.0, 6.5, 6.2831853 / max(k, 1e-5) / uMpp);
    float ph = dot(w.xy, q) - w.z * uTime;
    h += w.w * sin(ph) * lod;
    grad += w.w * cos(ph) * w.xy * lod;
    hMax += w.w * lod;
  }
  // Waves come in groups: their height swells and fades over a few hundred metres.
  float groups = 0.5 + 1.0 * vnoise(q / 520.0 + vec2(uTime * 0.02, 0.0));
  grad *= groups;
  h *= groups;
  // Fine texture that the eye expects on water, a few pixels across at any zoom.
  vec2 r = q / (uMpp * 5.0) + vec2(uTime * 0.21, -uTime * 0.17);
  float e = 0.35;
  vec2 fine = vec2(vnoise(r + vec2(e, 0.0)) - vnoise(r - vec2(e, 0.0)),
                   vnoise(r + vec2(0.0, e)) - vnoise(r - vec2(0.0, e)));
  vec3 n = normalize(vec3(-grad * 1.3 - fine * 0.13, 1.0));

  float day = 1.0 - uNight;
  vec3 L = normalize(uSun);
  float sunUp = smoothstep(-0.04, 0.12, L.z);
  float lowSun = sunUp * (1.0 - smoothstep(0.06, 0.4, L.z));

  // The water: Sargasso blue by day, moonlit blue by night, in slow patches the current carries.
  vec3 deep = mix(vec3(0.014, 0.040, 0.085), vec3(0.012, 0.135, 0.255), day);
  vec3 lift = mix(vec3(0.040, 0.095, 0.165), vec3(0.030, 0.300, 0.420), day);
  float m = 0.6 * vnoise(q / 460.0) + 0.4 * vnoise(q / 170.0 + 7.3);
  vec3 col = mix(deep, lift, 0.12 + 0.24 * m);
  // Light through the thin tops of the waves.
  col += lift * 0.30 * clamp(h / hMax, 0.0, 1.0) * (0.35 + 0.65 * day);

  // Slopes that face the sun are lighter; slopes catch the sky.
  vec3 toSun = normalize(vec3(L.xy, max(L.z, 0.25)));
  col *= 0.82 + 0.36 * clamp(dot(n, toSun), 0.0, 1.0) * sunUp;
  vec3 sky = mix(vec3(0.10, 0.14, 0.24), vec3(0.50, 0.68, 0.86), day);
  sky = mix(sky, vec3(0.95, 0.55, 0.34), lowSun * 0.75);
  col = mix(col, sky, clamp((1.0 - n.z) * 5.0, 0.0, 0.32));

  // Sun glitter, seen from straight above.
  vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
  float nh = max(dot(n, H), 0.0);
  float glint = (pow(nh, 420.0) * 2.4 + pow(nh, 40.0) * 0.07) * sunUp;
  col += mix(vec3(1.0, 0.97, 0.88), vec3(1.0, 0.72, 0.45), lowSun) * glint;
  // Moonlight at night: a cool, faint glitter, so the waves can still be seen.
  vec3 M = normalize(uMoon.xyz);
  float nm = max(dot(n, normalize(M + vec3(0.0, 0.0, 1.0))), 0.0);
  col += vec3(0.62, 0.72, 0.95) * (pow(nm, 260.0) * 1.1 + pow(nm, 30.0) * 0.05) * uMoon.w;

  // Whitecaps: some cells of the water hold one, each breaking and fading over 7 s,
  // so that on average uFoam of the sea is white. Stretched along the wind.
  vec2 wd = uWindDir;
  vec2 wr = vec2(wd.y, -wd.x);
  vec2 a = vec2(dot(q, wd), dot(q, wr));
  float cellM = 34.0;
  vec2 cell = floor(a / cellM);
  vec2 cf = fract(a / cellM);
  float r1 = hash21(cell);
  float r2 = hash21(cell + 17.31);
  float r3 = hash21(cell + 41.73);
  float life = fract(uTime / 7.0 + r2);
  float env = smoothstep(0.0, 0.08, life) * (1.0 - smoothstep(0.25, 1.0, life));
  vec2 c = vec2(0.5) + (vec2(r2, r3) - 0.5) * 0.36;
  vec2 d = (cf - c) / vec2(0.30, 0.13);
  d.x *= d.x > 0.0 ? 0.45 : 1.0;          // a streak of foam trails downwind of the break
  float blob = 1.0 - smoothstep(0.35, 1.0, length(d) / (0.55 + 0.45 * env));
  float p = clamp(uFoam / 0.045, 0.0, 1.0);
  float foam = step(r1, p) * blob * env * (0.55 + 0.45 * vnoise(q / 3.5 + uTime * 0.3));
  // Too small to draw close to the lower zooms: spread it as a sheen instead.
  float foamLod = smoothstep(1.6, 4.5, cellM * 0.3 / uMpp);
  foam = foam * foamLod;
  vec3 foamCol = mix(vec3(0.30, 0.95, 0.80) * 0.55, vec3(0.94, 0.97, 1.0), day);
  col = mix(col, foamCol, clamp(foam, 0.0, 1.0));
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
  } catch {
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
  for (const name of ['uRes', 'uMpp', 'uCentre', 'uFlow', 'uTime', 'uWaves', 'uWindDir', 'uFoam',
    'uSun', 'uMoon', 'uNight', 'uFade', 'uLand', 'uLandBox', 'uHasLand']) {
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

  const waveData = new Float32Array(WAVES * 4);

  return {
    setLand(mask, w, h, box) {
      gl.bindTexture(gl.TEXTURE_2D, land);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, w, h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, mask);
      landBox = box;
    },

    clearLand() { landBox = null; },

    /** Whether the context was lost (a driver reset); the layer then falls back. */
    lost() { return gl.isContextLost(); },

    draw(p) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      waveData.fill(0);
      p.waves.slice(0, WAVES).forEach((w, i) => waveData.set([w.kx, w.ky, w.omega, w.amp], i * 4));
      gl.uniform2f(u.uRes, canvas.width, canvas.height);
      gl.uniform1f(u.uMpp, p.mpp);
      gl.uniform2f(u.uCentre, p.centre[0], p.centre[1]);
      gl.uniform2f(u.uFlow, p.flow[0], p.flow[1]);
      gl.uniform1f(u.uTime, p.time);
      gl.uniform4fv(u.uWaves, waveData);
      gl.uniform2f(u.uWindDir, p.windDir[0], p.windDir[1]);
      gl.uniform1f(u.uFoam, p.foam);
      gl.uniform3f(u.uSun, p.sun[0], p.sun[1], p.sun[2]);
      const moon = p.moon ?? [0, 0, 1, 0];
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
