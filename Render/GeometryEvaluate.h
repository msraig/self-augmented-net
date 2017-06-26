#pragma once
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <helper_math.h>


__device__ void vectorUnproj(float2 pointScreen, float screenWidth, float screenHeight, float* matProj, float* matView, float3* out_start_point, float3* out_end_point)
{
	float mVPInv[16];
	float mVP[16];
	mat4Mul(matView, matProj, mVP);
	mat4Inv(mVP, mVPInv);

	float4 p0 = make_float4(2.0f * (pointScreen.x - screenWidth) / screenWidth - 1.0f, 1.0f - 2.0f * (pointScreen.y - screenHeight) / screenHeight, 0.0f, 1.0f);
	float4 p1 = make_float4(2.0f * (pointScreen.x - screenWidth) / screenWidth - 1.0f, 1.0f - 2.0f * (pointScreen.y - screenHeight) / screenHeight, 1.0f, 1.0f);

	//TODO: mul p0 and p1
	float4 p0_out = mul(p0, mVPInv);
	float4 p1_out = mul(p1, mVPInv);

	out_start_point->x = p0_out.x / p0_out.w;
	out_start_point->y = p0_out.y / p0_out.w;
	out_start_point->z = p0_out.z / p0_out.w;

	out_end_point->x = p1_out.x / p1_out.w;
	out_end_point->y = p1_out.y / p1_out.w;
	out_end_point->z = p1_out.z / p1_out.w;
}


__device__ void get_geometry_ball(float2* tex, float3* wNrm, float3* wPos, int px, int py, float screenWidth, float screenHeight, float* matProj, float* matView)
{
	//tex->x = 0.5; tex->y = 0.5;
	wPos->x = ((px+0.5f) - 0.5 * screenWidth) / (0.5 * screenWidth);
	wPos->y = -(0.5 * screenHeight - (py+0.5f)) / (0.5 * screenHeight);
	float tag = 1.0 - (wPos->x * wPos->x + wPos->y * wPos->y);
	if(tag < 0)
	{
		tex->x = -1.0; tex->y = -1.0;
		wNrm->x = 0.0; wNrm->y = 0.0; wNrm->z = 1.0;
		wPos->x = 0.0; wPos->y = 0.0; wNrm->z = 0.0;			
	}
	else
	{
		wPos->z = sqrt(tag);
		wNrm->x = wPos->x; wNrm->y = wPos->y; wNrm->z = wPos->z;
		tex->x = 0.5; tex->y = 0.5;
	}
}

__device__ void get_geometry_plane(float2* tex, float3* wNrm, float3* wPos, int px, int py, float screenWidth, float screenHeight, float* matProj, float* matView)
{
//Assume front view and every pixel is on plane.
	tex->x = (px + 0.5f) / screenWidth; tex->y = (py + 0.5f) / screenHeight;
	wNrm->x = 0.0; wNrm->y = 0.0; wNrm->z = 1.0;
	wPos->x = ((px+0.5f) - 0.5 * screenWidth) / (0.5 * screenWidth);
	wPos->y = -(0.5 * screenHeight - (py+0.5f)) / (0.5 * screenHeight);
	wPos->z = 0.0;
}

