#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

// BT.709 Cb and Cr in limited range, sampled between four source pixels so linear filtering averages them
void main() {
    vec3 rgb = texture2D(tex, v_coords).rgb;
    float cb = dot(rgb, vec3(-0.1146, -0.3854, 0.5));
    float cr = dot(rgb, vec3(0.5, -0.4542, -0.0458));
    gl_FragColor = vec4(vec2(cb, cr) * 224.0 / 255.0 + 128.0 / 255.0, 0.0, 1.0);
}
