//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "Advanced/XUSGPostprocess.h"

class PostprocessX :
    public XUSG::Postprocess_Impl
{
public:
	enum ExPipelineIndex : uint8_t
	{
		CIRCLE_OF_CONF,
		BILATERAL_DOF_DOWN,
		BILATERAL_DOF_UP,
		TEMPORAL_AA,

		NUM_EX_PIPELINE
	};

    PostprocessX(XUSG::API api = XUSG::API::DIRECTX_12);
    virtual ~PostprocessX();

	bool Init(const XUSG::Device* pDevice, const XUSG::ShaderLib::sptr& shaderLib,
		const XUSG::Graphics::PipelineLib::sptr& graphicsPipelineLib,
		const XUSG::Compute::PipelineLib::sptr& computePipelineLib,
		const XUSG::PipelineLayoutLib::sptr& pipelineLayoutLib,
		const XUSG::DescriptorTableLib::sptr& descriptorTableLib,
		XUSG::Format hdrFormat, XUSG::Format ldrFormat);
	bool ChangeWindowSize(const XUSG::Device* pDevice, const XUSG::Texture* pReference);
	bool SetDepth(const XUSG::DepthStencil* pDepth);

	void SetTime(double time);
	void DepthOfField(XUSG::CommandList* pCommandList, XUSG::Texture* pSource);
	void TemporalAA(XUSG::CommandList* pCommandList, XUSG::RenderTarget** ppDsts, XUSG::Texture** ppSrcs,
		const XUSG::DescriptorTable& uavTable, const XUSG::DescriptorTable& srvTable, uint8_t numUAVs, uint8_t numSRVs);

	XUSG::DescriptorTable CreateTemporalAASRVTable(const XUSG::Descriptor& srvCurrent, const XUSG::Descriptor& srvPrevious,
		const XUSG::Descriptor& srvVelocity, const XUSG::Descriptor& srvMasks, const XUSG::Descriptor& srvMeta);

protected:
	// Compute shaders
	enum ExComputeShader : uint8_t
	{
		CS_COC_GEN = XUSG::CS_LUM_ADAPT + 1,
		CS_DOF_DOWN,
		CS_DOF_UP,
		CS_TEMPORAL_AA
	};

	bool createPipelineLayouts();
	bool createPipelines(XUSG::Format hdrFormat, XUSG::Format ldrFormat);
	bool createDescriptorTables();

	void circleOfConfusion(XUSG::CommandList* pCommandList);
	void bilateralDown(XUSG::CommandList* pCommandList, XUSG::Texture* pSource);
	void bilateralUp(XUSG::CommandList* pCommandList);

	XUSG::PipelineLayout		m_exPipelineLayouts[NUM_EX_PIPELINE];
	XUSG::Pipeline				m_exPipelines[NUM_EX_PIPELINE];

	XUSG::Texture::uptr			m_circleOfConf;
	XUSG::Texture::uptr			m_sourceMip;
	XUSG::RenderTarget::uptr	m_filtered;

	XUSG::DescriptorTable		m_srvDepthTable;
	XUSG::DescriptorTable		m_uavCoCTable;
	//XUSG::DescriptorTable		m_uavTAATable;
	std::vector<XUSG::DescriptorTable> m_uavDoFDownTables;
	std::vector<XUSG::DescriptorTable> m_uavDoFUpTables;
	std::vector<XUSG::DescriptorTable> m_srvDoFTables;
	std::vector<XUSG::DescriptorTable> m_srvDoFUpTables;

	double m_time = 0.0;
};
