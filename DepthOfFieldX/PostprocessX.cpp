//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

//#include "Advanced/XUSGAdvanced.h"
#include "PostprocessX.h"

#define _TONE_MAPPED_BLIT_

using namespace std;
using namespace DirectX;
using namespace XUSG;

struct CBCamCoCParams
{
	float CocScale;
	float CocBias;
	float CocToImageSpace;
};

PostprocessX::PostprocessX(API api) :
	Postprocess_Impl(api)
{
}

PostprocessX::~PostprocessX()
{
}

bool PostprocessX::Init(const Device* pDevice, const ShaderLib::sptr& shaderLib,
	const Graphics::PipelineLib::sptr& graphicsPipelineLib,
	const Compute::PipelineLib::sptr& computePipelineLib,
	const PipelineLayoutLib::sptr& pipelineLayoutLib,
	const DescriptorTableLib::sptr& descriptorTableLib,
	Format hdrFormat, Format ldrFormat)
{
	XUSG_N_RETURN(Postprocess_Impl::Init(pDevice, shaderLib, graphicsPipelineLib, computePipelineLib,
		pipelineLayoutLib, descriptorTableLib, hdrFormat, ldrFormat), false);

	XUSG_N_RETURN(createPipelineLayouts(), false);
	XUSG_N_RETURN(createPipelines(hdrFormat, ldrFormat), false);

	return true;
}

bool PostprocessX::ChangeWindowSize(const Device* pDevice, const Texture* pReference)
{
	XUSG_N_RETURN(Postprocess_Impl::ChangeWindowSize(pDevice, pReference), false);

	// Create resources and pipelines
	const auto width = static_cast<uint32_t>(pReference->GetWidth());
	const auto height = pReference->GetHeight();
	m_numMipLevels = Texture::CalculateMipLevels(width, height);

	m_circleOfConf = Texture::MakeUnique();
	XUSG_N_RETURN(m_circleOfConf->Create(pDevice, width, height, Format::R32_FLOAT, 1,
		ResourceFlag::ALLOW_UNORDERED_ACCESS, m_numMipLevels, 1, false, MemoryFlag::NONE,
		L"CircleOfConfusion"), false);

	m_source = Texture::MakeUnique();
	XUSG_N_RETURN(m_source->Create(pDevice, width, height, pReference->GetFormat(), 1,
		ResourceFlag::ALLOW_UNORDERED_ACCESS, m_numMipLevels, 1, false, MemoryFlag::NONE,
		L"SourceImage"), false);

	m_filteredImage = Texture::MakeUnique();
	XUSG_N_RETURN(m_filteredImage->Create(pDevice, width, height, pReference->GetFormat(),
		1, ResourceFlag::ALLOW_UNORDERED_ACCESS, m_numMipLevels, 1, false, MemoryFlag::NONE,
		L"FilteredImage"), false);

	m_filteredCoC = Texture::MakeUnique();
	XUSG_N_RETURN(m_filteredCoC->Create(pDevice, width, height, Format::R32_FLOAT,
		1, ResourceFlag::ALLOW_UNORDERED_ACCESS, m_numMipLevels, 1, false, MemoryFlag::NONE,
		L"FilteredCoC"), false);

	return createDescriptorTables();
}

bool PostprocessX::SetDepth(DepthStencil* pDepth)
{
	m_pDepth = pDepth;
	const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
	descriptorTable->SetDescriptors(0, 1, &pDepth->GetSRV());
	XUSG_X_RETURN(m_srvDepthTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);

	return true;
}

void PostprocessX::SetTime(double time)
{
	m_time = time;
}

void PostprocessX::DepthOfField(XUSG::CommandList* pCommandList, Texture* pSceneColor,
	const DescriptorTable& uavTable, const DescriptorTable& srvTable)
{
#ifdef _TONE_MAPPED_BLIT_
	toneMappedBlit(pCommandList, m_source.get(), pSceneColor, m_uavDoFDownTables[0], srvTable, false);
#else
	ResourceBarrier barriers[2];
	auto numBarriers = m_source->SetBarrier(barriers, ResourceState::COPY_DEST, 0, 0);
	numBarriers = pSceneColor->SetBarrier(barriers, ResourceState::COPY_SOURCE, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	{
		TextureCopyLocation dst(m_sources->get(), 0);
		TextureCopyLocation src(pSceneColor, 0);
		pCommandList->CopyTextureRegion(dst, 0, 0, 0, src);
	}
#endif

	circleOfConfusion(pCommandList);
	bilateralDown(pCommandList, pSceneColor);
	bilateralUp(pCommandList);

#ifdef _TONE_MAPPED_BLIT_
	toneMappedBlit(pCommandList, pSceneColor, m_filteredImage.get(), uavTable, m_srvDoFUpTables[0], true);
#else
	numBarriers = m_filteredImage->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE | ResourceState::COPY_SOURCE, 0, 0);
	numBarriers = pSceneColor->SetBarrier(barriers, ResourceState::COPY_DEST, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	{
		TextureCopyLocation dst(pSceneColor, 0);
		TextureCopyLocation src(m_filteredImages->get(), 0);
		pCommandList->CopyTextureRegion(dst, 0, 0, 0, src);
	}
#endif
}

void PostprocessX::TemporalAA(CommandList* pCommandList, uint8_t numUAVs, Texture** ppDsts,
	const DescriptorTable& uavTable, uint8_t numSRVs, Texture** ppSrcs, const DescriptorTable& srvTable)
{
	// Set barriers
	vector<ResourceBarrier> barriers(numUAVs + numSRVs + 1);
	auto numBarriers = m_pVelocity->SetBarrier(barriers.data(), ResourceState::NON_PIXEL_SHADER_RESOURCE);
	for (uint8_t i = 0; i < numUAVs; ++i)
		numBarriers = ppDsts[i]->SetBarrier(barriers.data(), ResourceState::UNORDERED_ACCESS, numBarriers);
	for (uint8_t i = 0; i < numSRVs; ++i)
		numBarriers = ppSrcs[i]->SetBarrier(barriers.data(), ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
	pCommandList->Barrier(numBarriers, barriers.data());

	// Set pipeline layout and descriptor tables
	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[TEMPORAL_AA]);
	pCommandList->SetComputeDescriptorTable(TEXTURES, srvTable);
	pCommandList->SetComputeDescriptorTable(IMMUTABLE, m_cbvTables[CBV_IMMUTABLE]);
	pCommandList->SetComputeDescriptorTable(2, uavTable);

	// Set pipeline
	pCommandList->SetPipelineState(m_exPipelines[TEMPORAL_AA]);

	// Dispath
	const auto width = static_cast<uint32_t>((*ppDsts)->GetWidth());
	const auto height = (*ppDsts)->GetHeight();
	pCommandList->Dispatch(XUSG_DIV_UP(width, 8), XUSG_DIV_UP(height, 8), 1);

	numBarriers = ppDsts[0]->SetBarrier(barriers.data(), ResourceState::ALL_SHADER_RESOURCE);
	pCommandList->Barrier(numBarriers, barriers.data());
}

DescriptorTable PostprocessX::CreateTAASrvTable(const Descriptor& srvCurrent, const Descriptor& srvPrevious,
	const Texture* pVelocity, const Descriptor& srvShadeAmt, const Descriptor& srvMeta)
{
	m_pVelocity = const_cast<Texture*>(pVelocity);
	const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
	const Descriptor descriptors[] = { srvCurrent, srvPrevious, pVelocity->GetSRV(), srvShadeAmt, srvMeta, m_circleOfConf->GetSRV() };
	descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);

	return descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get());
}

bool PostprocessX::createPipelineLayouts()
{
	const auto pSampler = m_descriptorTableLib->GetSampler(LINEAR_CLAMP);

	// CoC generation
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_COC_GEN, L"CSCircleOfConf.cso"), false);
	{
		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Resources
		utilPipelineLayout->SetRange(0, DescriptorType::UAV, 1, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::SRV, 1, 0);
		utilPipelineLayout->SetConstants(2, 3, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(1, Shader::Stage::CS);

		XUSG_X_RETURN(m_exPipelineLayouts[CIRCLE_OF_CONF], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"CoCGenerationLayout"), false);
	}

	// DoF down sampling
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_DOF_DOWN, L"CSBilateralDown.cso"), false);
	{
		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Resources
		utilPipelineLayout->SetRange(0, DescriptorType::UAV, 2, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::SRV, 2, 0);
		utilPipelineLayout->SetConstants(2, 1, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(1, Shader::Stage::CS);

		// Sampler
		utilPipelineLayout->SetStaticSamplers(&pSampler, 1, 0);

		XUSG_X_RETURN(m_exPipelineLayouts[BILATERAL_DOF_DOWN], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"DoFDownLayout"), false);
	}

	// DoF up sampling
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_DOF_UP, L"CSBilateralUp.cso"), false);
	{
		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Resources
		utilPipelineLayout->SetRange(0, DescriptorType::UAV, 2, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::SRV, 2, 0);
		utilPipelineLayout->SetRange(2, DescriptorType::SRV, 2, 2);
		utilPipelineLayout->SetRange(3, DescriptorType::SRV, 2, 4);
		utilPipelineLayout->SetConstants(4, 1, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(1, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(2, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(3, Shader::Stage::CS);

		XUSG_X_RETURN(m_exPipelineLayouts[BILATERAL_DOF_UP], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"DoFUpLayout"), false);
	}

	// Temporal AA
	{
		auto cbImmutable = 0u;
		auto txImage = 0u;
		auto txHistory = txImage + 1;
		auto txVelocity = txHistory + 1;
		auto txMasks = txVelocity + 1;
		auto txHistMeta = txMasks + 1;
		auto txCoc = txHistMeta + 1;
		auto smpLinearClamp = 0u;

		// Load shader
		XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_TEMPORAL_AA, L"CSTemporalAA.cso"), false);

		// Get pixel shader slots
		auto reflector = m_shaderLib->GetReflector(Shader::Stage::CS, TEMPORAL_AA);
		if (reflector && reflector->IsValid())
		{
			// Get constant buffer slot
			cbImmutable = reflector->GetResourceBindingPointByName("cbImmutable", cbImmutable);

			// Get shader resource slots
			txImage = reflector->GetResourceBindingPointByName("g_txImage", txImage);
			txHistory = reflector->GetResourceBindingPointByName("g_txHistory", txHistory);
			txVelocity = reflector->GetResourceBindingPointByName("g_txVelocity", txVelocity);
			txMasks = reflector->GetResourceBindingPointByName("g_txMask", txMasks);
			txHistMeta = reflector->GetResourceBindingPointByName("g_txHistMeta", txHistMeta);

			// Get sampler slot
			smpLinearClamp = reflector->GetResourceBindingPointByName("g_smpLinear", smpLinearClamp);
		}

		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Constant buffers
		utilPipelineLayout->SetRange(IMMUTABLE, DescriptorType::CBV, 1, cbImmutable,
			0, DescriptorFlag::DATA_STATIC);

		// Textures
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txImage);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txHistory);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txVelocity);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txMasks);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txHistMeta);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txCoc);

		utilPipelineLayout->SetRange(2, DescriptorType::UAV, 2, 0);

		// Sampler
		utilPipelineLayout->SetStaticSamplers(&pSampler, 1, smpLinearClamp);

		XUSG_X_RETURN(m_exPipelineLayouts[TEMPORAL_AA], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"TemporalAACLayout"), false);
	}

	// Tone-mapped blit
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_TM_BLIT, L"CSTMBlit.cso"), false);
	{
		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Texture
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::UAV, 1, 0);
		utilPipelineLayout->SetShaderStage(TEXTURES, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(1, Shader::Stage::CS);

		XUSG_X_RETURN(m_exPipelineLayouts[TM_BLIT], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"ToneMappedBlitLayout"), false);
	}

	// Inverse tone-mapped blit
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_ITM_BLIT, L"CSITMBlit.cso"), false);
	m_exPipelineLayouts[ITM_BLIT] = m_exPipelineLayouts[TM_BLIT];

	// Tone mapping
	{
		auto txImage = 0u;
		auto roLogLum = txImage + 1;
		auto txCoC = roLogLum + 1;

		// Load shader
		XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::PS, PS_TONE_MAP, L"PSToneMap.cso"), false);

		// Get pixel shader slots
		auto reflector = m_shaderLib->GetReflector(Shader::Stage::PS, PS_TONE_MAP);
		if (reflector && reflector->IsValid())
		{
			// Get shader resource slots
			txImage = reflector->GetResourceBindingPointByName("g_txImage", txImage);
			roLogLum = reflector->GetResourceBindingPointByName("g_roLogLum", roLogLum);
		}

		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Textures
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txImage);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, roLogLum);
		utilPipelineLayout->SetRange(TEXTURES, DescriptorType::SRV, 1, txCoC);
		utilPipelineLayout->SetShaderStage(TEXTURES, Shader::Stage::PS);

		XUSG_X_RETURN(m_pipelineLayouts[TONE_MAP], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"ToneMappingLayout"), false);
	}

	return true;
}

bool PostprocessX::createPipelines(Format hdrFormat, Format ldrFormat)
{
	// CoC generation
	{
		const auto state = Compute::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_exPipelineLayouts[CIRCLE_OF_CONF]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, CS_COC_GEN));
		XUSG_X_RETURN(m_exPipelines[CIRCLE_OF_CONF], state->GetPipeline(m_computePipelineLib.get(), L"CocGeneration"), false);
	}

	// DoF down sampling
	{
		const auto state = Compute::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_exPipelineLayouts[BILATERAL_DOF_DOWN]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, CS_DOF_DOWN));
		XUSG_X_RETURN(m_exPipelines[BILATERAL_DOF_DOWN], state->GetPipeline(m_computePipelineLib.get(), L"DoFDown"), false);
	}

	// DoF up sampling
	{
		const auto state = Compute::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_exPipelineLayouts[BILATERAL_DOF_UP]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, CS_DOF_UP));
		XUSG_X_RETURN(m_exPipelines[BILATERAL_DOF_UP], state->GetPipeline(m_computePipelineLib.get(), L"DoFUp"), false);
	}

	// Temporal AA
	{
		const auto state = Compute::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_exPipelineLayouts[TEMPORAL_AA]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, CS_TEMPORAL_AA));
		XUSG_X_RETURN(m_exPipelines[TEMPORAL_AA], state->GetPipeline(m_computePipelineLib.get(), L"TemporalAAC"), false);
	}

	// Tone-mapped blit
	{
		const auto state = Compute::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_exPipelineLayouts[TM_BLIT]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, CS_TM_BLIT));
		XUSG_X_RETURN(m_exPipelines[TM_BLIT], state->GetPipeline(m_computePipelineLib.get(), L"ToneMappedBlit"), false);
	}

	// Inverse tone-mapped blit
	{
		const auto state = Compute::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_exPipelineLayouts[ITM_BLIT]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, CS_ITM_BLIT));
		XUSG_X_RETURN(m_exPipelines[ITM_BLIT], state->GetPipeline(m_computePipelineLib.get(), L"InverseToneMappedBlit"), false);
	}

	// Tone mapping
	{
		// Get tone-mapping pipeline
		const auto state = Graphics::State::MakeUnique(m_api);
		state->SetPipelineLayout(m_pipelineLayouts[TONE_MAP]);
		state->SetShader(Shader::Stage::VS, m_shaderLib->GetShader(Shader::Stage::VS, VS_SCREEN_QUAD));
		state->SetShader(Shader::Stage::PS, m_shaderLib->GetShader(Shader::Stage::PS, PS_TONE_MAP));
		state->OMSetNumRenderTargets(1);
		state->OMSetRTVFormat(0, ldrFormat);
		XUSG_X_RETURN(m_pipelines[TONE_MAP], state->GetPipeline(m_graphicsPipelineLib.get(), L"ToneMapping"), false);
	}

	return true;
}

bool PostprocessX::createDescriptorTables()
{
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		descriptorTable->SetDescriptors(0, 1, &m_circleOfConf->GetUAV());
		XUSG_X_RETURN(m_uavCoCTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_uavDoFDownTables.resize(m_numMipLevels);
	for (uint8_t i = 0; i < m_numMipLevels; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_source->GetUAV(i),
			m_circleOfConf->GetUAV(i)
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_uavDoFDownTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_uavDoFUpTables.resize(m_numMipLevels);
	for (uint8_t i = 0; i < m_numMipLevels; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_filteredImage->GetUAV(i),
			m_filteredCoC->GetUAV(i)
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_uavDoFUpTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_srvDoFTables.resize(m_numMipLevels);
	for (uint8_t i = 0; i < m_numMipLevels; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_source->GetSRV(i, true),
			m_circleOfConf->GetSRV(i, true)
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_srvDoFTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_srvDoFUpTables.resize(m_numMipLevels);
	for (uint8_t i = 0; i < m_numMipLevels; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			i + 1 < m_numMipLevels ? m_filteredImage->GetSRV(i, true) : m_source->GetSRV(i, true),
			i + 1 < m_numMipLevels ? m_filteredCoC->GetSRV(i, true) : m_circleOfConf->GetSRV(i, true)
			//m_circleOfConfs[i]->GetSRV()
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_srvDoFUpTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] = { m_postImage->GetSRV(), m_avgLum->GetSRV(), m_circleOfConf->GetSRV() };
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_uavSrvTables[SRV_COLOR_AVG_LUM], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	return true;
}

void PostprocessX::circleOfConfusion(CommandList* pCommandList)
{
	const auto weekly = [](double t)
	{
		const float s = static_cast<float>(sin(t));

		return s * s * s * s * s;
	};

	const auto width = static_cast<uint32_t>(m_circleOfConf->GetWidth());
	const auto height = m_circleOfConf->GetHeight();

	// Update camera DoF parameters
	CBCamCoCParams camCoCParams;
	{
		const auto aperture = 0.05f;
		const auto focalLength = 0.25f;
		const auto planeInFocus = 20.0f;// +weekly(m_time * 0.5) * 8.0f;
		const auto imageHeight = 0.0625f;

		const auto af = aperture * focalLength;
		const auto d = (planeInFocus - focalLength) * XUSG_zNear;

		camCoCParams.CocScale = af * planeInFocus * (XUSG_zFar - XUSG_zNear) / (d * XUSG_zFar);
		camCoCParams.CocBias = af * (XUSG_zNear - planeInFocus) / d;
		camCoCParams.CocToImageSpace = height / imageHeight;
	}

	ResourceBarrier barriers[2];
	auto numBarriers = m_circleOfConf->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
	numBarriers = m_pDepth->SetBarrier(barriers, ResourceState::ALL_SHADER_RESOURCE, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[CIRCLE_OF_CONF]);
	pCommandList->SetComputeDescriptorTable(0, m_uavCoCTable);
	pCommandList->SetComputeDescriptorTable(1, m_srvDepthTable);
	pCommandList->SetCompute32BitConstants(2, 3, &camCoCParams);

	pCommandList->SetPipelineState(m_exPipelines[CIRCLE_OF_CONF]);

	pCommandList->Dispatch(XUSG_DIV_UP(width, 8), XUSG_DIV_UP(height, 8), 1);
}

void PostprocessX::bilateralDown(CommandList* pCommandList, Texture* pSource)
{
	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[BILATERAL_DOF_DOWN]);
	pCommandList->SetPipelineState(m_exPipelines[BILATERAL_DOF_DOWN]);

	const auto width = static_cast<uint32_t>(m_source->GetWidth());
	const auto height = m_source->GetHeight();

	ResourceBarrier barriers[3];
	for (uint8_t i = 1; i < m_numMipLevels; ++i)
	{
		const uint8_t f = i - 1;
		auto numBarriers = m_source->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS, 0, i);
		numBarriers = m_source->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers, f);
		numBarriers = m_circleOfConf->SetBarrier(barriers, ResourceState::ALL_SHADER_RESOURCE, numBarriers, f);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeDescriptorTable(0, m_uavDoFDownTables[i]);
		pCommandList->SetComputeDescriptorTable(1, m_srvDoFTables[f]);
		pCommandList->SetCompute32BitConstant(2, i);

		// Dispatch grid
		const auto threadsX = (max)(width >> i, 1u);
		const auto threadsY = (max)(height >> i, 1u);
		pCommandList->Dispatch(XUSG_DIV_UP(threadsX, 8), XUSG_DIV_UP(threadsY, 8), 1);
	}
}

void PostprocessX::bilateralUp(CommandList* pCommandList)
{
	const uint8_t numPasses = m_numMipLevels - 1;

	ResourceBarrier barriers[4];
	auto numBarriers = m_source->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, 0, numPasses);
	numBarriers = m_circleOfConf->SetBarrier(barriers, ResourceState::ALL_SHADER_RESOURCE, numBarriers, numPasses);

	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[BILATERAL_DOF_UP]);
	pCommandList->SetPipelineState(m_exPipelines[BILATERAL_DOF_UP]);

	const auto width = static_cast<uint32_t>(m_filteredImage->GetWidth());
	const auto height = m_filteredImage->GetHeight();

	for (uint8_t i = 0; i < numPasses; ++i)
	{
		const auto c = numPasses - i;
		const auto l = c - 1;

		numBarriers = m_filteredImage->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS, numBarriers, l);
		numBarriers = m_filteredImage->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE |
			ResourceState::COPY_SOURCE, numBarriers, c);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeDescriptorTable(0, m_uavDoFUpTables[l]);
		pCommandList->SetComputeDescriptorTable(1, m_srvDoFUpTables[c]);
		pCommandList->SetComputeDescriptorTable(2, m_srvDoFTables[l]);
		pCommandList->SetComputeDescriptorTable(3, m_srvDoFTables[c]);
		pCommandList->SetCompute32BitConstant(4, l);

		// Dispatch grid
		const auto threadsX = (max)(width >> l, 1u);
		const auto threadsY = (max)(height >> l, 1u);
		pCommandList->Dispatch(XUSG_DIV_UP(threadsX, 8), XUSG_DIV_UP(threadsY, 8), 1);
		numBarriers = 0;
	}
}

void PostprocessX::toneMappedBlit(CommandList* pCommandList, Texture* pDst, Texture* pSrc,
	const DescriptorTable& uavTable, const DescriptorTable& srvTable, bool inverse)
{
	ResourceBarrier barriers[2];
	auto numBarriers = pDst->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS, 0, 0);
	numBarriers = pSrc->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers, 0);
	pCommandList->Barrier(numBarriers, barriers);

	const auto pipeIdx = inverse ? ITM_BLIT : TM_BLIT;
	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[pipeIdx]);
	pCommandList->SetPipelineState(m_exPipelines[pipeIdx]);

	pCommandList->SetComputeDescriptorTable(TEXTURES, srvTable);
	pCommandList->SetComputeDescriptorTable(1, uavTable);

	const auto width = static_cast<uint32_t>(pDst->GetWidth());
	const auto height = pDst->GetHeight();
	pCommandList->Dispatch(XUSG_DIV_UP(width, 8), XUSG_DIV_UP(height, 8), 1);
}
