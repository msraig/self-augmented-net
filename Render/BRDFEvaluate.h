#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#define MATH_PI 3.14159265358979f

#define SPE_LOD_OFFSET 1.f
#define DIFF_LOD_OFFSET 2.f

// Hammersley sequence generator
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
#define NUM_LIGHT_MIPMAP 9

texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube0;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube1;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube2;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube3;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube4;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube5;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube6;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube7;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> texCube8;


__device__ float3 samplefloat3Fromfloat4(float4 input)
{
	return make_float3(input.x, input.y, input.z);
}

__device__ float3 samplefloat3Fromfloat(float input)
{
	return make_float3(input, input, input);
}

__device__ float4 texCubeMipmap(float x, float y, float z, float lod)
{
	int lodLeft = int(lod);
	float wLeft = 1.0 - (lod - lodLeft);
	float wRight = 1.0 - wLeft;

	float4 sampled = make_float4(0.0f);
	switch(lodLeft)
	{
		case 0:
		sampled = wLeft * texCubemap(texCube0, x, y, z) + wRight * texCubemap(texCube1, x, y, z);
		break;
		case 1:
		sampled = wLeft * texCubemap(texCube1, x, y, z) + wRight * texCubemap(texCube2, x, y, z);
		break;
		case 2:
		sampled = wLeft * texCubemap(texCube2, x, y, z) + wRight * texCubemap(texCube3, x, y, z);
		break;
		case 3:
		sampled = wLeft * texCubemap(texCube3, x, y, z) + wRight * texCubemap(texCube4, x, y, z);
		break;
		case 4://case 5:case 6:case 7: case 8:
		sampled = wLeft * texCubemap(texCube4, x, y, z) + wRight * texCubemap(texCube5, x, y, z);
		break;
		case 5:
		sampled = wLeft * texCubemap(texCube5, x, y, z) + wRight * texCubemap(texCube6, x, y, z);
		break;
		case 6:
		sampled = wLeft * texCubemap(texCube6, x, y, z) + wRight * texCubemap(texCube7, x, y, z);
		break;
		case 7:
		sampled = wLeft * texCubemap(texCube7, x, y, z) + wRight * texCubemap(texCube8, x, y, z);
		break;
		case 8:
		sampled = texCubemap(texCube8, x, y, z);// + wRight * texCubemap(texCube9, x, y, z);
		break;
	}	
//	sampled = texCubemap(texCube0, x, y, z);
//	sampled = texCubemap(texCube0, x, y, z);
	//ampled = wLeft * texCubemapLod(texCube[lodLeft], x, y, z, float(lodLeft)) + wRight * texCubemapLod(texCube[lodRight], x, y, z, float(lodRight));
	return sampled;
}


__device__ float radicalInverse_VdC(uint bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

__device__ float2 Hammersley(uint i, uint NumSamples)
{
	return make_float2(float(i) / float(NumSamples), radicalInverse_VdC(i));
}

// PDFs
// Beckmann
// http://blog.selfshadow.com/publications/s2012-shading-course/
// PDF_L = PDF_H /(4 * VoH)
__device__ float PDF_Beckmann_H(float roughness, float NoH)
{
	float e = exp((NoH * NoH - 1) / (NoH * NoH * roughness * roughness));

	// beckmann * NoH
	return e / (MATH_PI * roughness * roughness * NoH * NoH * NoH);
}

// Diffuse
__device__ float PDF_Diffuse(float NoL)
{
	return NoL / MATH_PI;
}

//Ref: Microfacet Models for Refraction through Rough Surfaces, EGSR 2007
//Beckmann
__device__ float NDF_Beckmann(float3 M, float3 N, float roughness)
{
	float MoN = dot(M, N);
	float CosThetaM = clamp(MoN, 0.0, 1.0);
	if (CosThetaM > 0)
	{
		float roughness_2 = roughness * roughness;
		float CosThetaM_2 = CosThetaM * CosThetaM;
		float CosThetaM_4 = CosThetaM_2 * CosThetaM_2;
		float TanThetaM_2 = (1.0 - CosThetaM_2) / CosThetaM_2;

		return exp(-TanThetaM_2 / roughness_2) / (MATH_PI * roughness_2 * CosThetaM_4);
	}
	else
		return 0;
}

// Importance Sampling Functions
// Beckmann
__device__ float3 ImportanceSampleBeckmann(float2 Xi, float roughness, float3 N)
{
	float Phi = 2 * MATH_PI * Xi.x;
	float CosTheta = sqrt(1.f / (1 - roughness * roughness * log(1 - Xi.y)));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);
	float3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;
	float3 UpVector = (abs(N.z) < 0.5f) ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
	float3 TangentX = normalize(cross(UpVector, N));
	float3 TangentY = cross(N, TangentX);
	// Tangent to world space
	return TangentX * H.x + TangentY * H.y + N * H.z;
}

// Schlick-Smith Geometric term
//http://blog.selfshadow.com/publications/s2012-shading-course/mcauley/s2012_pbs_farcry3_notes_v2.pdf
__device__ float G_SchlickSmith(float roughness, float NoL, float NoV)
{
	float a = roughness * sqrt(2.f / MATH_PI);
	float visInv = (NoL * (1.f - a) + a) * (NoV *(1.f - a) + a);
	return NoL * NoV / visInv;
}
//Cook-Torrance Geometric term
__device__ float G_CookTorrance(float NoL, float NoV, float NoH, float VoH)
{
	float shad1 = (2.0f * NoH * NoV) / VoH;
	float shad2 = (2.0f * NoH * NoL) / VoH;
	return min(1.0f, min(shad1, shad2));
}

// Diffuse
__device__ float3 ImportanceSampleDiffuse(float2 Xi, float3 N)
{
	float Phi = 2 * MATH_PI * Xi.x;
	float CosTheta = sqrt(1 - Xi.y);
	float SinTheta = sqrt(1 - CosTheta * CosTheta);
	float3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;

	float3 UpVector = (abs(N.z) < 0.5f) ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
	//float3 UpVector = normalize((1 - abs(N.z + N.x + N.y)) * float3(0, 0, 1) + 0.5f * abs(N.z + N.x + N.y) * float3(1, 0, 0));
	float3 TangentX = normalize(cross(UpVector, N));
	float3 TangentY = normalize(cross(N, TangentX));
	// Tangent to world space
	return TangentX * H.x + TangentY * H.y + N * H.z;
}



//Shading Eval functions
__device__ float3 EvalDiffusePointLight(float3 L, float3 V, float3 N)
{
	float d = clamp(dot(L, N), 0.0, 1.0) / MATH_PI;
	return make_float3(d);
}

__device__ float3 EvalSpecularPointLight(float3 L, float3 V, float3 N, float roughness)
{
	float NoL = clamp(dot(L, N), 0.0, 1.0);
	float NoV = clamp(dot(N, V), 0.0, 1.0);
	if (NoL > 1e-6 && NoV > 1e-6)
	{
		float3 H = normalize(0.5 *(L + V));
		float NoH = clamp(dot(N, H), 0.0, 1.0) + 1e-10f;
		float VoH = clamp(dot(V, H), 0.0, 1.0) + 1e-10f;

		float D = 0;
		D = NDF_Beckmann(H, N, roughness);
		float G = G_CookTorrance(NoL, NoV, NoH, VoH);
		return make_float3(G * D / (4.0 * NoV));
	}
	else
		return make_float3(0.0);
}



__device__ float3 EvalDiffuseEnvLight(float3 N, uint nSamples, uint nCubeRes)
{
	float3 DiffuseLighting = make_float3(0);

	// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
	// ignore distortion
	const float solidAng_pixel = 4 * MATH_PI / (nCubeRes * nCubeRes * 6);
//	nSamples = 128;//
//	float avg_l = make_float3(0);

	for (uint i = 0; i < nSamples; i++)
	{
		float2 Xi = Hammersley(i, nSamples);
		float3 L = ImportanceSampleDiffuse(Xi, N);
//		avg_l += L;
		float NoL = clamp(dot(N, L), 0.0, 1.0);
		if (NoL > 0.0f)
		{
			float solidAng_sample = 1.f / (nSamples * PDF_Diffuse(NoL));
			float lod = min(NUM_LIGHT_MIPMAP - 1.0f, (max(0.0f, 0.5f * log2(solidAng_sample / solidAng_pixel)) + DIFF_LOD_OFFSET));
//			lod = 0;
			float3 SampleColor = samplefloat3Fromfloat4(texCubeMipmap(L.x, L.y, L.z, lod));
			//float3 SampleColor = samplefloat3Fromfloat4(texCubemapLod(texCube, L.x, L.y, L.z, lod));
			DiffuseLighting += SampleColor;
		}
	}
//	avg_l /= nSamples;
	return DiffuseLighting / nSamples;//make_float3(avg_l.z, avg_l.y, avg_l.x);//
}


__device__ float3 EvalSpecularEnvLight(float roughness, float3 N, float3 V, uint nSamples, uint nCubeRes)
{
	// float3 L = 2 * dot(V, N) * N - V;
	float3 SpecularLighting = make_float3(0);
	// float3 SampleColor = samplefloat3Fromfloat4(texCubeMipmap(L.x, L.y, L.z, 0));
	// return SampleColor;//make_float3(L.z, L.y, L.x);

	// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
	// ignore distortion
	const float solidAng_pixel = 4 * MATH_PI / (nCubeRes * nCubeRes * 6);
	

	for (uint i = 0; i < nSamples; i++)
	{
		float2 Xi = Hammersley(i, nSamples);
		//float3 H = make_float3(0.0f, 0.0f, 1.0f);
		float3 H = ImportanceSampleBeckmann(Xi, roughness, N);

		float3 L = 2 * dot(V, H) * H - V;
		float NoV = clamp(dot(N, V), 0.0, 1.0) + 1e-10f;
		float NoL = clamp(dot(N, L), 0.0, 1.0);
		float NoH = clamp(dot(N, H), 0.0, 1.0) + 1e-10f;
		float VoH = clamp(dot(V, H), 0.0, 1.0) + 1e-10f;
		if (NoL > 0)
		{
			// http://blog.selfshadow.com/publications/s2012-shading-course/
			// PDF_L = PDF_H /(4 * VoH)
			float solidAng_sample = solidAng_pixel;
			solidAng_sample = 4.f * VoH / (nSamples * PDF_Beckmann_H(roughness, NoH));
			float lod = min(NUM_LIGHT_MIPMAP - 1.0f, (max(0.0f, 0.5f * log2(solidAng_sample / solidAng_pixel)) + SPE_LOD_OFFSET));
			float3 SampleColor = samplefloat3Fromfloat4(texCubeMipmap(L.x, L.y, L.z, lod));
			float G = G_CookTorrance(NoL, NoV, NoH, VoH);//G_SchlickSmith(roughness, NoL, NoV);
			SpecularLighting += SampleColor * G * VoH / (NoH * NoV);
		}
	}
	return SpecularLighting / nSamples;//make_float3(avgLod / nSamples);//SpecularLighting / nSamples;////SpecularLighting / nSamples;//make_float3(avgLod / nSamples);//make_float3(avgLod.z / nSamples, avgLod.y / nSamples, avgLod.x / nSamples);//make_float3(avgLod / nSamples); //SpecularLighting / nSamples;
}