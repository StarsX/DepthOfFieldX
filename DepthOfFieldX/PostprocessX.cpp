//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

//#include "Advanced/XUSGAdvanced.h"
#include "PostprocessX.h"

#define _TONE_MAPPED_BLIT_
#define BASIS_KERNEL_SIZE 3

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
	//const auto numMips = CalculateMipLevels(width, height);
	m_numLayers = PYRAMID_LAYERS;

	XMUINT2 layerSize(width, height);
	for (uint8_t i = 0; i < PYRAMID_LAYERS; ++i)
	{
		if (layerSize.x == 0 || layerSize.y == 0)
		{
			m_numLayers = i;
			break;
		}

		m_circleOfConfs[i] = Texture::MakeUnique();
		XUSG_N_RETURN(m_circleOfConfs[i]->Create(pDevice, layerSize.x, layerSize.y, Format::R32_FLOAT,
			1, ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, false, MemoryFlag::NONE,
			(L"CircleOfConfusion" + to_wstring(i)).c_str()), false);

		m_sources[i] = Texture::MakeUnique();
		XUSG_N_RETURN(m_sources[i]->Create(pDevice, layerSize.x, layerSize.y, pReference->GetFormat(),
			1, ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, false, MemoryFlag::NONE,
			(L"SourceLayer" + to_wstring(i)).c_str()), false);

		m_filteredImages[i] = Texture::MakeUnique();
		XUSG_N_RETURN(m_filteredImages[i]->Create(pDevice, layerSize.x, layerSize.y, pReference->GetFormat(),
			1, ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, false, MemoryFlag::NONE,
			(L"FilteredImage" + to_wstring(i)).c_str()), false);

		m_filteredCoCs[i] = Texture::MakeUnique();
		XUSG_N_RETURN(m_filteredCoCs[i]->Create(pDevice, layerSize.x, layerSize.y, Format::R32_FLOAT,
			1, ResourceFlag::ALLOW_UNORDERED_ACCESS, 1, 1, false, MemoryFlag::NONE,
			(L"FilteredCoC" + to_wstring(i)).c_str()), false);

		layerSize.x /= BASIS_KERNEL_SIZE;
		layerSize.y /= BASIS_KERNEL_SIZE;
	}

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
	toneMappedBlit(pCommandList, m_sources->get(), pSceneColor, m_uavDoFDownTables[0], srvTable, false);
#else
	ResourceBarrier barriers[2];
	auto numBarriers = m_sources[0]->SetBarrier(barriers, ResourceState::COPY_DEST);
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
	toneMappedBlit(pCommandList, pSceneColor, m_filteredImages->get(), uavTable, m_srvDoFUpTables[0], true);
#else
	numBarriers = m_filteredImages[0]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE | ResourceState::COPY_SOURCE, 0, 0);
	numBarriers = pSceneColor->SetBarrier(barriers, ResourceState::COPY_DEST, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	{
		TextureCopyLocation dst(pSceneColor, 0);
		TextureCopyLocation src(m_filteredImages->get(), 0);
		pCommandList->CopyTextureRegion(dst, 0, 0, 0, src);
	}
#endif
}

void PostprocessX::TemporalAA(CommandList* pCommandList, RenderTarget** ppDsts, Texture** ppSrcs,
	const DescriptorTable& uavTable, const DescriptorTable& srvTable, uint8_t numUAVs, uint8_t numSRVs)
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
	const Descriptor descriptors[] = { srvCurrent, srvPrevious, pVelocity->GetSRV(), srvShadeAmt, srvMeta, m_circleOfConfs[0]->GetSRV() };
	descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);

	return descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get());
}

bool PostprocessX::createPipelineLayouts()
{
	assert(BASIS_KERNEL_SIZE == 2 || BASIS_KERNEL_SIZE == 3);
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
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_DOF_DOWN, BASIS_KERNEL_SIZE == 2 ? L"CSBilateralDown.cso" : L"CSBilateralDown3x3.cso"), false);
	{
		// Pipeline layout utility
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique(m_api);

		// Resources
		utilPipelineLayout->SetRange(0, DescriptorType::UAV, 2, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::SRV, 2, 0);
		utilPipelineLayout->SetConstants(2, 1, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(1, Shader::Stage::CS);

		// Samplers
		utilPipelineLayout->SetStaticSamplers(&pSampler, 1, 0);

		XUSG_X_RETURN(m_exPipelineLayouts[BILATERAL_DOF_DOWN], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"DoFDownLayout"), false);
	}

	// DoF up sampling
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, CS_DOF_UP, BASIS_KERNEL_SIZE == 2 ? L"CSBilateralUp.cso" : L"CSBilateralUp3x3.cso"), false);
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

		// Samplers
		utilPipelineLayout->SetStaticSamplers(&pSampler, 1, 0);

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

	return true;
}

bool PostprocessX::createDescriptorTables()
{
	const auto& numLayers = m_numLayers;

	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		descriptorTable->SetDescriptors(0, 1, &m_circleOfConfs[0]->GetUAV());
		XUSG_X_RETURN(m_uavCoCTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_uavDoFDownTables.resize(numLayers);
	for (uint8_t i = 0; i < numLayers; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_sources[i]->GetUAV(),
			m_circleOfConfs[i]->GetUAV()
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_uavDoFDownTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_uavDoFUpTables.resize(numLayers);
	for (uint8_t i = 0; i < numLayers; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_filteredImages[i]->GetUAV(),
			m_filteredCoCs[i]->GetUAV()
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_uavDoFUpTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_srvDoFTables.resize(numLayers);
	for (uint8_t i = 0; i < numLayers; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_sources[i]->GetSRV(),
			m_circleOfConfs[i]->GetSRV()
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_srvDoFTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_srvDoFUpTables.resize(numLayers);
	for (uint8_t i = 0; i < numLayers; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			i + 1 < numLayers ? m_filteredImages[i]->GetSRV() : m_sources[i]->GetSRV(),
			i + 1 < numLayers ? m_filteredCoCs[i]->GetSRV() : m_circleOfConfs[i]->GetSRV()
			//m_circleOfConfs[i]->GetSRV()
		};
		descriptorTable->SetDescriptors(0, static_cast<uint32_t>(size(descriptors)), descriptors);
		XUSG_X_RETURN(m_srvDoFUpTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	return true;;
}

void PostprocessX::circleOfConfusion(CommandList* pCommandList)
{
	const auto weekly = [](double t)
	{
		const float s = static_cast<float>(sin(t));

		return s * s * s * s * s;
	};

	const auto width = static_cast<uint32_t>(m_circleOfConfs[0]->GetWidth());
	const auto height = m_circleOfConfs[0]->GetHeight();

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
	auto numBarriers = m_circleOfConfs[0]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
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
	const auto& numLayers = m_numLayers;

	ResourceBarrier barriers[3];
	for (uint8_t i = 1; i < numLayers; ++i)
	{
		auto numBarriers = m_sources[i]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
		numBarriers = m_sources[i - 1]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
		numBarriers = m_circleOfConfs[i - 1]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeDescriptorTable(0, m_uavDoFDownTables[i]);
		pCommandList->SetComputeDescriptorTable(1, m_srvDoFTables[i - 1]);
		pCommandList->SetCompute32BitConstant(2, i);

		// Dispatch grid
		const auto threadsX = static_cast<uint32_t>(m_sources[i]->GetWidth());
		const auto threadsY = m_sources[i]->GetHeight();
		pCommandList->Dispatch(XUSG_DIV_UP(threadsX, 8), XUSG_DIV_UP(threadsY, 8), 1);
	}
}

void PostprocessX::bilateralUp(CommandList* pCommandList)
{
	const auto& numLayers = m_numLayers;

	ResourceBarrier barriers[4];
	auto numBarriers = m_sources[numLayers - 1]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE);
	numBarriers = m_circleOfConfs[numLayers - 1]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers);

	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[BILATERAL_DOF_UP]);
	pCommandList->SetPipelineState(m_exPipelines[BILATERAL_DOF_UP]);

	const uint8_t numPasses = numLayers - 1;
	for (uint8_t i = 0; i < numPasses; ++i)
	{
		const auto c = numPasses - i;
		const auto l = c - 1;

		numBarriers = m_filteredImages[l]->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS, numBarriers);
		numBarriers = m_filteredImages[c]->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE | ResourceState::COPY_SOURCE, numBarriers);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeDescriptorTable(0, m_uavDoFUpTables[l]);
		pCommandList->SetComputeDescriptorTable(1, m_srvDoFUpTables[c]);
		pCommandList->SetComputeDescriptorTable(2, m_srvDoFTables[l]);
		pCommandList->SetComputeDescriptorTable(3, m_srvDoFTables[c]);
		pCommandList->SetCompute32BitConstant(4, l);

		// Dispatch grid
		const auto threadsX = static_cast<uint32_t>(m_filteredImages[l]->GetWidth());
		const auto threadsY = m_filteredImages[l]->GetHeight();
		pCommandList->Dispatch(XUSG_DIV_UP(threadsX, 8), XUSG_DIV_UP(threadsY, 8), 1);
		numBarriers = 0;
	}
}

void PostprocessX::toneMappedBlit(CommandList* pCommandList, Texture* pDst, Texture* pSrc,
	const DescriptorTable& uavTable, const DescriptorTable& srvTable, bool inverse)
{
	ResourceBarrier barriers[2];
	auto numBarriers = pDst->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
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
