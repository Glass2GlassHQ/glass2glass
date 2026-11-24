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

// BT.709 luma in limited range
void main() {
    vec3 rgb = texture2D(tex, v_coords).rgb;
    float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    gl_FragColor = vec4(luma * 219.0 / 255.0 + 16.0 / 255.0, 0.0, 0.0, 1.0);
}
