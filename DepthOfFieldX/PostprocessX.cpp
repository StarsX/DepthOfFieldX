//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

//#include "Advanced/XUSGAdvanced.h"
#include "PostprocessX.h"

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
	const auto numMips = CalculateMipLevels(width, height);

	m_circleOfConf = Texture::MakeUnique();
	m_circleOfConf->Create(pDevice, width, height, Format::R32_FLOAT,
		1, ResourceFlag::ALLOW_UNORDERED_ACCESS, numMips, 1, false,
		MemoryFlag::NONE, L"CircleOfConfusion");

	m_sourceMip = Texture::MakeUnique();
	m_sourceMip->Create(pDevice, width, height, pReference->GetFormat(),
		1, ResourceFlag::ALLOW_UNORDERED_ACCESS, numMips, 1, false,
		MemoryFlag::NONE, L"SourceMipMap");

	m_filtered = RenderTarget::MakeUnique();
	m_filtered->Create(pDevice, width, height, pReference->GetFormat(),
		1, ResourceFlag::ALLOW_UNORDERED_ACCESS, numMips, 1, nullptr,
		false, MemoryFlag::NONE, L"FilteredImage");

	return createDescriptorTables();
}

bool PostprocessX::SetDepth(const DepthStencil* pDepth)
{
	const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
	descriptorTable->SetDescriptors(0, 1, &pDepth->GetSRV());
	XUSG_X_RETURN(m_srvDepthTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);

	return true;
}

void PostprocessX::SetTime(double time)
{
	m_time = time;
}

void PostprocessX::DepthOfField(XUSG::CommandList* pCommandList, XUSG::Texture* pSource)
{
	ResourceBarrier barriers[2];
	auto numBarriers = m_sourceMip->SetBarrier(barriers, ResourceState::COPY_DEST);
	numBarriers = pSource->SetBarrier(barriers, ResourceState::COPY_SOURCE | ResourceState::PIXEL_SHADER_RESOURCE, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	{
		TextureCopyLocation dst(m_sourceMip.get(), 0);
		TextureCopyLocation src(pSource, 0);
		pCommandList->CopyTextureRegion(dst, 0, 0, 0, src);
	}

	numBarriers = m_sourceMip->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS);
	pCommandList->Barrier(numBarriers, barriers);

	circleOfConfusion(pCommandList);
	bilateralDown(pCommandList, pSource);
	bilateralUp(pCommandList);

	numBarriers = m_filtered->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE | ResourceState::COPY_SOURCE, 0, 0);
	numBarriers = pSource->SetBarrier(barriers, ResourceState::COPY_DEST, numBarriers);
	pCommandList->Barrier(numBarriers, barriers);

	{
		TextureCopyLocation dst(pSource, 0);
		TextureCopyLocation src(m_filtered.get(), 0);
		pCommandList->CopyTextureRegion(dst, 0, 0, 0, src);
	}
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

		// Samplers
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
		utilPipelineLayout->SetRange(0, DescriptorType::UAV, 1, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::SRV, 2, 0);
		utilPipelineLayout->SetRange(2, DescriptorType::SRV, 2, 2);
		utilPipelineLayout->SetConstants(3, 1, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(1, Shader::Stage::CS);
		utilPipelineLayout->SetShaderStage(2, Shader::Stage::CS);

		// Samplers
		utilPipelineLayout->SetStaticSamplers(&pSampler, 1, 0);

		XUSG_X_RETURN(m_exPipelineLayouts[BILATERAL_DOF_UP], utilPipelineLayout->GetPipelineLayout(m_pipelineLayoutLib.get(),
			PipelineLayoutFlag::NONE, L"DoFDownLayout"), false);
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
		XUSG_X_RETURN(m_exPipelines[BILATERAL_DOF_UP], state->GetPipeline(m_computePipelineLib.get(), L"DoFDown"), false);
	}

	return true;
}

bool PostprocessX::createDescriptorTables()
{
	const auto numMips = m_circleOfConf->GetNumMips();

	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		descriptorTable->SetDescriptors(0, 1, &m_circleOfConf->GetUAV());
		XUSG_X_RETURN(m_uavCoCTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_uavDoFDownTables.resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_sourceMip->GetUAV(i),
			m_circleOfConf->GetUAV(i)
		};
		descriptorTable->SetDescriptors(0, 2, descriptors);
		XUSG_X_RETURN(m_uavDoFDownTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_uavDoFUpTables.resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		descriptorTable->SetDescriptors(0, 1, &m_filtered->GetUAV(i));
		XUSG_X_RETURN(m_uavDoFUpTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_srvDoFTables.resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_sourceMip->GetSRVLevel(i),
			m_circleOfConf->GetSRVLevel(i)
		};
		descriptorTable->SetDescriptors(0, 2, descriptors);
		XUSG_X_RETURN(m_srvDoFTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	m_srvDoFUpTables.resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique(m_api);
		const Descriptor descriptors[] =
		{
			m_filtered->GetSRVLevel(i),
			m_circleOfConf->GetSRVLevel(i)
		};
		descriptorTable->SetDescriptors(0, 2, descriptors);
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

	const auto width = static_cast<uint32_t>(m_circleOfConf->GetWidth());
	const auto height = m_circleOfConf->GetHeight();

	// Update camera DoF parameters
	CBCamCoCParams camCoCParams;
	{
		const auto aperture = 0.125f;
		const auto focalLength = 0.25f;
		const auto planeInFocus = 20.0f;// +weekly(m_time * 0.5) * 8.0f;
		const auto imageHeight = 0.0625f;

		const auto af = aperture * focalLength;
		const auto d = (planeInFocus - focalLength) * XUSG_zNear;

		camCoCParams.CocScale = af * planeInFocus * (XUSG_zFar - XUSG_zNear) / (d * XUSG_zFar);
		camCoCParams.CocBias = af * (XUSG_zNear - planeInFocus) / d;
		camCoCParams.CocToImageSpace = height / imageHeight;
	}

	ResourceBarrier barrier;
	const auto numBarriers = m_circleOfConf->SetBarrier(&barrier, ResourceState::UNORDERED_ACCESS);
	pCommandList->Barrier(numBarriers, &barrier);

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
	const auto numMips = m_circleOfConf->GetNumMips();

	const auto width = static_cast<uint32_t>(m_circleOfConf->GetWidth());
	const auto height = m_circleOfConf->GetHeight();

	ResourceBarrier barriers[2];
	for (uint8_t i = 1; i < numMips; ++i)
	{
		auto numBarriers = m_sourceMip->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, 0, i - 1);
		numBarriers = m_circleOfConf->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers, i - 1);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeDescriptorTable(0, m_uavDoFDownTables[i]);
		pCommandList->SetComputeDescriptorTable(1, m_srvDoFTables[i - 1]);
		pCommandList->SetCompute32BitConstant(2, i);

		// Dispatch grid
		const auto threadsX = (max)(width >> i, 1u);
		const auto threadsY = (max)(height >> i, 1u);
		pCommandList->Dispatch(XUSG_DIV_UP(threadsX, 8), XUSG_DIV_UP(threadsY, 8), 1);
	}
}

void PostprocessX::bilateralUp(CommandList* pCommandList)
{
	const auto numMips = m_filtered->GetNumMips();

	ResourceBarrier barriers[4];
	auto numBarriers = m_sourceMip->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, 0, numMips - 1);
	numBarriers = m_circleOfConf->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE, numBarriers, numMips - 1);

	pCommandList->SetComputePipelineLayout(m_exPipelineLayouts[BILATERAL_DOF_UP]);
	pCommandList->SetPipelineState(m_exPipelines[BILATERAL_DOF_UP]);

	const auto width = static_cast<uint32_t>(m_filtered->GetWidth());
	const auto height = m_filtered->GetHeight();

	const uint8_t numPasses = numMips - 1;
	for (uint8_t i = 0; i < numPasses; ++i)
	{
		const auto c = numPasses - i;
		const auto level = c - 1;

		numBarriers = m_filtered->SetBarrier(barriers, ResourceState::UNORDERED_ACCESS, numBarriers, level);
		numBarriers = m_filtered->SetBarrier(barriers, ResourceState::NON_PIXEL_SHADER_RESOURCE | ResourceState::COPY_SOURCE, numBarriers, c);
		pCommandList->Barrier(numBarriers, barriers);

		pCommandList->SetComputeDescriptorTable(0, m_uavDoFUpTables[level]);
		pCommandList->SetComputeDescriptorTable(1, m_srvDoFUpTables[c]);
		pCommandList->SetComputeDescriptorTable(2, m_srvDoFTables[level]);
		pCommandList->SetCompute32BitConstant(3, level);

		// Dispatch grid
		const auto threadsX = (max)(width >> level, 1u);
		const auto threadsY = (max)(height >> level, 1u);
		pCommandList->Dispatch(XUSG_DIV_UP(threadsX, 8), XUSG_DIV_UP(threadsY, 8), 1);
		numBarriers = 0;
	}
}
