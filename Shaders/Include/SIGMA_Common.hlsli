/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Misc

#define PackShadow( s )         Math::Sqrt01( s ) // must match "SIGMA_BackEnd_UnpackShadow"
#define IsLit( p )              ( p >= NRD_FP16_MAX )

float3 GetViewVector( float3 X, bool isViewSpace = false )
{
    return gOrthoMode == 0.0 ? normalize( -X ) : ( isViewSpace ? float3( 0, 0, -1 ) : gViewVectorWorld.xyz );
}

float GetKernelRadiusInPixels( float hitDist, float unprojectZ, float scale = 1.0 )
{
    float unclampedRadius = hitDist / unprojectZ;
    unclampedRadius *= scale;

    #if( SIGMA_5X5_BLUR_RADIUS_ESTIMATION_KERNEL == 1 )
        float minRadius = min( unclampedRadius, 2.0 );
    #else
        float minRadius = min( unclampedRadius, 1.0 );
    #endif

    return clamp( unclampedRadius, minRadius, SIGMA_MAX_PIXEL_RADIUS );
}

float AreBothLitOrUnlit( float penumbra1, float penumbra2 )
{
    bool NoL1 = penumbra1 == 0.0;
    bool NoL2 = penumbra2 == 0.0;

    return float( NoL1 == NoL2 );
}

// TODO: move code below to STL.hlsl

float2 FilterBicubic(float2 size, float2 uv, out float4 uv_10_00, out float4 uv_11_01)
{
    const float4 c1 = float4(  3.0,  0.0,  1.0, 4.0 );
    const float4 c2 = float4( -1.0,  3.0, -3.0, 1.0 );
    const float4 c3 = float4(  3.0, -6.0, -3.0, 0.0 );
    const float k = 1.0 / 6.0;

    float4 dxdy = -c1.zyyz / size.xyxy;

    float2 f = frac( uv.xy * size - 0.5 );
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float3 xw, yw;
    float4 phi;

    phi = k * ( c2.xyzw * f3.xxxx + c3.xyxw * f2.xxxx + c3.zwxw * f.xxxx + c1.zwzy );
    xw.xy = c2.ww + c2.wx * f.xx + c2.xw * phi.yw / ( phi.xz + phi.yw );
    xw.z = phi.x + phi.y;

    phi = k * ( c2.xyzw * f3.yyyy + c3.xyxw * f2.yyyy + c3.zwxw * f.yyyy + c1.zwzy );
    yw.xy = c2.ww + c2.wx * f.yy + c2.xw * phi.yw / ( phi.xz + phi.yw );
    yw.z = phi.x + phi.y;

    uv_10_00 = uv.xyxy + c2.wwxx * xw.xxyy * dxdy.xyxy;
    uv_11_01 = uv_10_00 + yw.xxxx * dxdy.zwzw;

    uv_10_00 -= yw.yyyy * dxdy.zwzw;

    return float2( yw.z, xw.z );
}

float2 TextureCubic(Texture2D<float2> tex, float2 uv)
{
    uint w, h;
    tex.GetDimensions( w, h );
    float2 size = float2( w, h );

    float4 uv_10_00, uv_11_01;
    float2 t = FilterBicubic( size, uv.xy, uv_10_00, uv_11_01 );

    float2 c00 = tex.SampleLevel( gLinearClamp, uv_10_00.zw, 0 );
    float2 c10 = tex.SampleLevel( gLinearClamp, uv_10_00.xy, 0 );
    float2 c01 = tex.SampleLevel( gLinearClamp, uv_11_01.zw, 0 );
    float2 c11 = tex.SampleLevel( gLinearClamp, uv_11_01.xy, 0 );

    c00 = lerp( c00, c01, t.x );
    c10 = lerp( c10, c11, t.x );

    return lerp( c00, c10, t.y );
}

void BicubicFilterNoCorners(
    float2 samplePos, float2 invResourceSize, bool useBicubic,
    Texture2D<float4> tex0, out float4 c0 )
{
    if( useBicubic )
    {
        float4 bilinearCustomWeights = 0;

        _BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights_Init;
        _BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights_Color( c0, tex0 );
    }
    else
        c0 = tex0.SampleLevel( gLinearClamp, samplePos * invResourceSize, 0 );
}

void BicubicFilterNoCorners(
    float2 samplePos, float2 invResourceSize, bool useBicubic,
    Texture2D<float> tex0, out float c0 )
{
    if( useBicubic )
    {
        float4 bilinearCustomWeights = 0;

        _BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights_Init;
        _BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights_Color( c0, tex0 );
    }
    else
        c0 = tex0.SampleLevel( gLinearClamp, samplePos * invResourceSize, 0 );
}
