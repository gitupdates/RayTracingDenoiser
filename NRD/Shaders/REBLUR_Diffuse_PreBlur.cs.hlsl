/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "BindingBridge.hlsl"
#include "NRD.hlsl"
#include "STL.hlsl"
#include "REBLUR_Config.hlsl"

NRI_RESOURCE( cbuffer, globalConstants, b, 0, 0 )
{
    REBLUR_DIFF_SHARED_CB_DATA;

    float4x4 gWorldToView;
    float4 gRotator;
    float4 gDiffHitDistParams;
    uint gDiffCheckerboard;
};

#include "NRD_Common.hlsl"
#include "REBLUR_Common.hlsl"

// Inputs
NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 0, 0 );
NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 1, 0 );
NRI_RESOURCE( Texture2D<float4>, gIn_Diff, t, 2, 0 );

// Outputs
NRI_RESOURCE( RWTexture2D<float>, gOut_ScaledViewZ, u, 0, 0 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 1, 0 );

void Preload( int2 sharedId, int2 globalId )
{
    uint2 globalIdUser = gRectOrigin + globalId;

    s_Normal_Roughness[ sharedId.y ][ sharedId.x ] = _NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ globalIdUser ] );
    s_ViewZ[ sharedId.y ][ sharedId.x ] = abs( gIn_ViewZ[ globalIdUser ] );
}

[numthreads( GROUP_X, GROUP_Y, 1 )]
void main( int2 threadId : SV_GroupThreadId, int2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    PRELOAD_INTO_SMEM;

    // Checkerboard
    bool2 hasData = true;
    uint2 checkerboardPixelPos = pixelPos.xx;
    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex );

    if( gDiffCheckerboard != 2 )
    {
        hasData.x = checkerboard == gDiffCheckerboard;
        checkerboardPixelPos.x >>= 1;
    }

    // Early out
    int2 smemPos = threadId + BORDER;
    float viewZ = s_ViewZ[ smemPos.y ][ smemPos.x ];

    float scaledViewZ = min( viewZ * NRD_FP16_VIEWZ_SCALE, NRD_FP16_MAX );
    gOut_ScaledViewZ[ pixelPos ] = scaledViewZ;

    [branch]
    if( viewZ > gInf )
        return;

    // Normal and roughness
    float4 normalAndRoughness = s_Normal_Roughness[ smemPos.y ][ smemPos.x ];
    float3 N = normalAndRoughness.xyz;
    float3 Nv = STL::Geometry::RotateVector( gWorldToView, N );
    float roughness = normalAndRoughness.w;

    // Shared data
    float3 Xv = STL::Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gIsOrtho );
    float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, Xv, Nv, viewZ );
    float4 rotator = GetBlurKernelRotation( REBLUR_PRE_BLUR_ROTATOR_MODE, pixelPos, gRotator, gFrameIndex );

    // Edge detection
    float edge = DetectEdge( N, smemPos );

    // Center data
    float4 diff = gIn_Diff[ gRectOrigin + uint2( checkerboardPixelPos.x, pixelPos.y ) ];

    int3 smemCheckerboardPos = smemPos.xyx + int3( -1, 0, 1 );
    float viewZ0 = s_ViewZ[ smemCheckerboardPos.y ][ smemCheckerboardPos.x ];
    float viewZ1 = s_ViewZ[ smemCheckerboardPos.y ][ smemCheckerboardPos.z ];
    float2 w = GetBilateralWeight( float2( viewZ0, viewZ1 ), viewZ );
    w *= STL::Math::PositiveRcp( w.x + w.y );

    int3 checkerboardPos = pixelPos.xyx + int3( -1, 0, 1 );
    checkerboardPos.xz >>= 1;
    checkerboardPos += gRectOrigin.xyx;

    float4 d0 = gIn_Diff[ checkerboardPos.xy ];
    float4 d1 = gIn_Diff[ checkerboardPos.zy ];
    if( !hasData.x )
    {
        diff *= saturate( 1.0 - w.x - w.y );
        diff += d0 * w.x + d1 * w.y;
    }

    float2 error = 0;

    // Spatial filtering
    #define REBLUR_SPATIAL_MODE REBLUR_PRE_BLUR

    #include "REBLUR_Common_DiffuseSpatialFilter.hlsl"
}