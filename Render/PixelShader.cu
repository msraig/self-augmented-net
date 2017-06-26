#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include "BRDFEvaluate.h"
#include "GeometryEvaluate.h"

#define MAX_LIGHT_COUNT 8
#define RENDER_IMAGE 0
#define RENDER_ALBEDO 1
#define RENDER_SPECALBEDO 2
#define RENDER_ROUGHNESS 3
#define RENDER_NORMAL 4
#define RENDER_MASK 5
#define RENDER_POS 6
#define RENDER_VIEW 7
#define RENDER_UV 8
#define RENDER_LIGHTDIR 9

texture<float4, cudaTextureType2D, cudaReadModeElementType> albedo;
texture<float4, cudaTextureType2D, cudaReadModeElementType> spec;
texture<float, cudaTextureType2D, cudaReadModeElementType> roughness;
texture<float4, cudaTextureType2D, cudaReadModeElementType> normal;

//First version: fixed object pos and viewpoint

extern "C" {

__device__ float3 reverse(float3 in)
{
	return make_float3(in.z, in.y, in.x);
}



__global__ void PS_Render_Plane_Point(float* output, int imgwidth, int imgheight, 
						  float* matProj, float* matView,
						  bool* lightStatus, float* lightIntensity, float* lightPos, int renderMode)
{
	float mVInv[16];
	mat4Inv(matView, mVInv);
	float4 tmpEye = mul(make_float4(0, 0, 0, 1), mVInv);
	float3 eyePos = make_float3(tmpEye.x, tmpEye.y, tmpEye.z);


	int px = threadIdx.x + blockIdx.x * blockDim.x;
	int py = threadIdx.y + blockIdx.y * blockDim.y;		

	if(px < imgwidth && py < imgheight)
	{
		int imgindex = py * imgwidth + px;

		float2 tex = make_float2(0);
		float3 wPos = make_float3(0);
		float3 wNrm = make_float3(0);

		get_geometry_plane(&tex, &wNrm, &wPos, px, py, imgwidth, imgheight, matProj, matView);

		float3 diffColor = samplefloat3Fromfloat4(tex2D(albedo, tex.x, tex.y));
		float3 speColor = samplefloat3Fromfloat4(tex2D(spec, tex.x, tex.y));
		float roughnessValue = tex2D(roughness, tex.x, tex.y);
		wNrm = reverse(samplefloat3Fromfloat4(tex2D(normal, tex.x, tex.y)));

		if(wNrm.x > 2.0f)
		{
			output[3*imgindex] = 0.0f;
			output[3*imgindex+1] = 0.0f;
			output[3*imgindex+2] = 0.0f;
			return;
		}

		wNrm = normalize(wNrm);

		float3 V = normalize(eyePos - wPos);

		float3 color = make_float3(0);
		if(renderMode == RENDER_IMAGE)
		{
			float3 diffuse = make_float3(0);
			float3 specular = make_float3(0);
			for(int i=0; i<MAX_LIGHT_COUNT; i++)
			{
				if(!lightStatus[i])
					continue;

				float4 lPos = make_float4(lightPos[4*i], lightPos[4*i+1], lightPos[4*i+2], lightPos[4*i+3]);
				float3 L = make_float3(0);
				if(lPos.w < 0.5)
					L = normalize(make_float3(lPos.x, lPos.y, lPos.z));
				else
					L = normalize(make_float3(lPos.x - wPos.x, lPos.y - wPos.y, lPos.z - wPos.z));
				diffuse = make_float3(lightIntensity[3*i], lightIntensity[3*i+1], lightIntensity[3*i+2]) * EvalDiffusePointLight(L, V, wNrm);
				color += diffColor * diffuse;
				specular = make_float3(lightIntensity[3*i], lightIntensity[3*i+1], lightIntensity[3*i+2]) * EvalSpecularPointLight(L, V, wNrm, roughnessValue);
				color += speColor * specular;			
			}
		}
		else if(renderMode == RENDER_ALBEDO)
		{
			color = diffColor;
		}	
		else if(renderMode == RENDER_SPECALBEDO)
		{
			color = speColor;
		}	
		else if(renderMode == RENDER_ROUGHNESS)
		{
			color = make_float3(roughnessValue);
		}	
		else if(renderMode == RENDER_NORMAL)
		{	
			color = reverse(wNrm);
		}	
		else if(renderMode == RENDER_MASK)
		{
			color = make_float3(1.0f);
		}	
		else if(renderMode == RENDER_POS)
		{
			color = reverse(wPos);
		}	
		else if(renderMode == RENDER_VIEW)
		{
			color = reverse(V);
		}
		else if(renderMode == RENDER_UV)
		{
			color = make_float3(0.0, tex.y, tex.x);
		}		
		else if(renderMode == RENDER_LIGHTDIR)
		{
			float4 lPos = make_float4(lightPos[0], lightPos[1], lightPos[2], lightPos[3]);
			float3 L = make_float3(0);
			if(lPos.w < 0.5)
				L = normalize(make_float3(lPos.x, lPos.y, lPos.z));
			else
				L = normalize(make_float3(lPos.x - wPos.x, lPos.y - wPos.y, lPos.z - wPos.z));										
		}

		output[3*imgindex] = color.x;
		output[3*imgindex+1] = color.y;
		output[3*imgindex+2] = color.z;
	}
}

__global__ void PS_Render_Sphere_Point(float* output, int imgwidth, int imgheight, 
						  float* matProj, float* matView,
						  bool* lightStatus, float* lightIntensity, float* lightPos, int renderMode)
{
	float mVInv[16];
	mat4Inv(matView, mVInv);
	float4 tmpEye = mul(make_float4(0, 0, 0, 1), mVInv);
	float3 eyePos = make_float3(tmpEye.x, tmpEye.y, tmpEye.z);
	

	int px = threadIdx.x + blockIdx.x * blockDim.x;
	int py = threadIdx.y + blockIdx.y * blockDim.y;		

	if(px < imgwidth && py < imgheight)
	{
		int imgindex = py * imgwidth + px;

		float2 tex = make_float2(0);
		float3 wPos = make_float3(0);
		float3 wNrm = make_float3(0);

		get_geometry_plane(&tex, &wNrm, &wPos, px, py, imgwidth, imgheight, matProj, matView);

		float3 diffColor = samplefloat3Fromfloat4(tex2D(albedo, tex.x, tex.y));
		float3 speColor = samplefloat3Fromfloat4(tex2D(spec, tex.x, tex.y));
		float roughnessValue = tex2D(roughness, tex.x, tex.y);
		wNrm = reverse(samplefloat3Fromfloat4(tex2D(normal, tex.x, tex.y)));

		if(wNrm.x > 2.0f)
		{
			output[3*imgindex] = 0.0f;
			output[3*imgindex+1] = 0.0f;
			output[3*imgindex+2] = 0.0f;
			return;
		}

		wNrm = normalize(wNrm);
		wPos = wNrm;

		float3 V = normalize(eyePos - wPos);

		float3 color = make_float3(0);
		if(renderMode == RENDER_IMAGE)
		{
			float3 diffuse = make_float3(0);
			float3 specular = make_float3(0);
			for(int i=0; i<MAX_LIGHT_COUNT; i++)
			{
				if(!lightStatus[i])
					continue;

				float4 lPos = make_float4(lightPos[4*i], lightPos[4*i+1], lightPos[4*i+2], lightPos[4*i+3]);
				float3 L = make_float3(0);
				if(lPos.w < 0.5)
					L = normalize(make_float3(lPos.x, lPos.y, lPos.z));
				else
					L = normalize(make_float3(lPos.x - wPos.x, lPos.y - wPos.y, lPos.z - wPos.z));
				diffuse = make_float3(lightIntensity[3*i], lightIntensity[3*i+1], lightIntensity[3*i+2]) * EvalDiffusePointLight(L, V, wNrm);
				color += diffColor * diffuse;
				specular = make_float3(lightIntensity[3*i], lightIntensity[3*i+1], lightIntensity[3*i+2]) * EvalSpecularPointLight(L, V, wNrm, roughnessValue);
				color += speColor * specular;			
			}
		}
		else if(renderMode == RENDER_ALBEDO)
		{
			color = diffColor;
		}	
		else if(renderMode == RENDER_SPECALBEDO)
		{
			color = speColor;
		}	
		else if(renderMode == RENDER_ROUGHNESS)
		{
			color = make_float3(roughnessValue);
		}	
		else if(renderMode == RENDER_NORMAL)
		{	
			color = reverse(wNrm);
		}	
		else if(renderMode == RENDER_MASK)
		{
			color = make_float3(1.0f);
		}	
		else if(renderMode == RENDER_POS)
		{
			color = reverse(wPos);
		}	
		else if(renderMode == RENDER_VIEW)
		{
			color = reverse(V);
		}
		else if(renderMode == RENDER_UV)
		{
			color = make_float3(0.0, tex.y, tex.x);
		}		
		else if(renderMode == RENDER_LIGHTDIR)
		{
			float4 lPos = make_float4(lightPos[0], lightPos[1], lightPos[2], lightPos[3]);
			float3 L = make_float3(0);
			if(lPos.w < 0.5)
				L = normalize(make_float3(lPos.x, lPos.y, lPos.z));
			else
				L = normalize(make_float3(lPos.x - wPos.x, lPos.y - wPos.y, lPos.z - wPos.z));										
		}

		output[3*imgindex] = color.x;
		output[3*imgindex+1] = color.y;
		output[3*imgindex+2] = color.z;
	}
}


__global__ void PS_Render_Plane_Env(float* output, int imgwidth, int imgheight,
						  float* matProj, float* matView, 
						  float* matrixLight, uint nCubeRes,
						  uint nDiffuseSample, uint nSpecSample, int renderMode)
{
	float mVInv[16];
	mat4Inv(matView, mVInv);
	float4 tmpEye = mul(make_float4(0, 0, 0, 1), mVInv);
	float3 eyePos = make_float3(tmpEye.x, tmpEye.y, tmpEye.z);


	int px = threadIdx.x + blockIdx.x * blockDim.x;
	int py = threadIdx.y + blockIdx.y * blockDim.y;		

	if(px < imgwidth && py < imgheight)
	{
		int imgindex = py * imgwidth + px;

		float2 tex = make_float2(0);
		float3 wPos = make_float3(0);
		float3 wNrm = make_float3(0);

		get_geometry_plane(&tex, &wNrm, &wPos, px, py, imgwidth, imgheight, matProj, matView);

		
		float3 diffColor = samplefloat3Fromfloat4(tex2D(albedo, tex.x, tex.y));
		float3 speColor = samplefloat3Fromfloat4(tex2D(spec, tex.x, tex.y));
		float roughnessValue = tex2D(roughness, tex.x, tex.y);
		wNrm = reverse(samplefloat3Fromfloat4(tex2D(normal, tex.x, tex.y)));

		if(wNrm.x > 2.0f)
		{
			output[3*imgindex] = 0.0f;
			output[3*imgindex+1] = 0.0f;
			output[3*imgindex+2] = 0.0f;
			return;
		}
		wNrm = normalize(wNrm);

		float3 V = normalize(eyePos - wPos);
		float3 color = make_float3(0);

		float4 lNrm = mul(make_float4(wNrm.x, wNrm.y, wNrm.z, 0), matrixLight);
		float3 lightSpaceNrm = make_float3(lNrm.x, lNrm.y, lNrm.z);


		float4 lView = mul(make_float4(V.x, V.y, V.z, 0), matrixLight);
		float3 lightSpaceView = make_float3(lView.x, lView.y, lView.z);

		if(renderMode == RENDER_IMAGE)
		{
			float3 diffuse = EvalDiffuseEnvLight(lightSpaceNrm, nDiffuseSample, nCubeRes);
			float3 spec = EvalSpecularEnvLight(roughnessValue, lightSpaceNrm, lightSpaceView, nSpecSample, nCubeRes);
			color = diffuse * diffColor + spec * speColor;
		}
		else if(renderMode == RENDER_ALBEDO)
		{
			color = diffColor;
		}	
		else if(renderMode == RENDER_SPECALBEDO)
		{
			color = speColor;
		}	
		else if(renderMode == RENDER_ROUGHNESS)
		{
			color = make_float3(roughnessValue);
		}	
		else if(renderMode == RENDER_NORMAL)
		{	
			color = reverse(0.5*(wNrm+1.0f));
		}	
		else if(renderMode == RENDER_MASK)
		{
			color = make_float3(1.0f);
		}	
		else if(renderMode == RENDER_POS)
		{
			color = reverse(wPos);
		}	
		else if(renderMode == RENDER_VIEW)
		{
			color = reverse(lightSpaceView);
//			color = make_float3(matrixLight[6]);
		}
		else if(renderMode == RENDER_UV)
		{
			color = make_float3(0.0, tex.y, tex.x);
		}		
		else if(renderMode == RENDER_LIGHTDIR)
		{
			float3 avgL = make_float3(0);
			for (uint i = 0; i < nSpecSample; i++)
			{
				float2 Xi = Hammersley(i, nSpecSample);
				float3 H = make_float3(0.0f, 0.0f, 1.0f);
				H = ImportanceSampleBeckmann(Xi, roughnessValue, lightSpaceNrm);

				float3 L = 2 * dot(lightSpaceView, H) * H - lightSpaceView;
				avgL += L;
			}
			color = reverse(avgL / nSpecSample);											
		}
		output[3*imgindex] = color.x;
		output[3*imgindex+1] = color.y;
		output[3*imgindex+2] = color.z;
	}	
}

__global__ void PS_Render_Sphere_Env(float* output, int imgwidth, int imgheight,
						  float* matProj, float* matView, 
						  float* matrixLight, uint nCubeRes,
						  uint nDiffuseSample, uint nSpecSample, int renderMode)
{
	float mVInv[16];
	mat4Inv(matView, mVInv);
	float4 tmpEye = mul(make_float4(0, 0, 0, 1), mVInv);
	float3 eyePos = make_float3(tmpEye.x, tmpEye.y, tmpEye.z);


	int px = threadIdx.x + blockIdx.x * blockDim.x;
	int py = threadIdx.y + blockIdx.y * blockDim.y;		

	if(px < imgwidth && py < imgheight)
	{
		int imgindex = py * imgwidth + px;

		float2 tex = make_float2(0);
		float3 wPos = make_float3(0);
		float3 wNrm = make_float3(0);

		get_geometry_plane(&tex, &wNrm, &wPos, px, py, imgwidth, imgheight, matProj, matView);

		
		float3 diffColor = samplefloat3Fromfloat4(tex2D(albedo, tex.x, tex.y));
		float3 speColor = samplefloat3Fromfloat4(tex2D(spec, tex.x, tex.y));
		float roughnessValue = tex2D(roughness, tex.x, tex.y);
		wNrm = reverse(samplefloat3Fromfloat4(tex2D(normal, tex.x, tex.y)));

		if(wNrm.x > 2.0f)
		{
			output[3*imgindex] = 0.0f;
			output[3*imgindex+1] = 0.0f;
			output[3*imgindex+2] = 0.0f;
			return;
		}
		wNrm = normalize(wNrm);
		wPos = wNrm;

		float3 V = normalize(eyePos - wPos);
		float3 color = make_float3(0);

		float4 lNrm = mul(make_float4(wNrm.x, wNrm.y, wNrm.z, 0), matrixLight);
		float3 lightSpaceNrm = make_float3(lNrm.x, lNrm.y, lNrm.z);

		float4 lView = mul(make_float4(V.x, V.y, V.z, 0), matrixLight);
		float3 lightSpaceView = make_float3(lView.x, lView.y, lView.z);

		if(renderMode == RENDER_IMAGE)
		{
			float3 diffuse = EvalDiffuseEnvLight(lightSpaceNrm, nDiffuseSample, nCubeRes);
			float3 spec = EvalSpecularEnvLight(roughnessValue, lightSpaceNrm, lightSpaceView, nSpecSample, nCubeRes);
			color = diffuse * diffColor + spec * speColor;
		}
		else if(renderMode == RENDER_ALBEDO)
		{
			color = diffColor;
		}	
		else if(renderMode == RENDER_SPECALBEDO)
		{
			color = speColor;
		}	
		else if(renderMode == RENDER_ROUGHNESS)
		{
			color = make_float3(roughnessValue);
		}	
		else if(renderMode == RENDER_NORMAL)
		{	
			color = reverse(0.5*(wNrm+1.0f));
		}	
		else if(renderMode == RENDER_MASK)
		{
			color = make_float3(1.0f);
		}	
		else if(renderMode == RENDER_POS)
		{
			color = reverse(wPos);
		}	
		else if(renderMode == RENDER_VIEW)
		{
			color = reverse(lightSpaceView);
		}
		else if(renderMode == RENDER_UV)
		{
			color = make_float3(0.0, tex.y, tex.x);
		}		
		else if(renderMode == RENDER_LIGHTDIR)
		{
			float3 avgL = make_float3(0);
			for (uint i = 0; i < nSpecSample; i++)
			{
				float2 Xi = Hammersley(i, nSpecSample);
				float3 H = make_float3(0.0f, 0.0f, 1.0f);
				H = ImportanceSampleBeckmann(Xi, roughnessValue, lightSpaceNrm);

				float3 L = 2 * dot(lightSpaceView, H) * H - lightSpaceView;
				avgL += L;
			}
			color = reverse(avgL / nSpecSample);											
		}
		output[3*imgindex] = color.x;
		output[3*imgindex+1] = color.y;
		output[3*imgindex+2] = color.z;
	}	
}


}