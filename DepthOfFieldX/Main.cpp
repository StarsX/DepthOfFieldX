//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "DepthOfFieldX.h"
#include "resource.h"

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	DepthOfFieldX depthOfFieldX(1280, 800, L"Depth of Field");

	const auto hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_DEPTHOFFIELDX));

	return Win32Application::Run(&depthOfFieldX, hInstance, nCmdShow, hIcon);
}
