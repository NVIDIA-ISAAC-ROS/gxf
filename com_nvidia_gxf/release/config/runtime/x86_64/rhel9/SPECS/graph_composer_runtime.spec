################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

Name:           graph_composer_runtime
Version:        4.1.0
Release:        1%{?dist}
Summary:        NVIDIA Graph Tools
License:        NVIDIA Proprietary
URL:            https://docs.nvidia.com/metropolis/deepstream/dev-guide/graphtools-docs/docs/text/GraphComposer_intro.html
Source:         graph_composer-4.1.0_el9_x86_64.tar.gz
Requires:       rsync
AutoReqProv:    no

%description
NVIDIA Graph Tools. Runtime only package for creating and running the GXF graphs.

%prep
%setup -n opt

%install
mkdir -p %{buildroot}/opt
cp -R * %{buildroot}/opt

%post
update-alternatives --install %{_libdir}/libgxf_core.so gxf_core /opt/nvidia/graph-composer/libgxf_core.so 50
update-alternatives --install %{_bindir}/registry registry /opt/nvidia/graph-composer/registry 50
update-alternatives --install %{_bindir}/container_builder container_builder /opt/nvidia/graph-composer/container_builder 50
update-alternatives --install %{_bindir}/gxe gxe /opt/nvidia/graph-composer/gxe 50
update-alternatives --install %{_bindir}/gxf_cli gxf_cli /opt/nvidia/graph-composer/gxf_cli 50
update-alternatives --install %{_bindir}/gxf_server gxf_server /opt/nvidia/graph-composer/gxf_server 50
rm -rf /var/tmp/gxf
ldconfig

%postun
update-alternatives --remove gxf_core /opt/nvidia/graph-composer/libgxf_core.so
update-alternatives --remove registry /opt/nvidia/graph-composer/registry
update-alternatives --remove container_builder /opt/nvidia/graph-composer/container_builder
update-alternatives --remove gxe /opt/nvidia/graph-composer/gxe
update-alternatives --remove gxf_cli /opt/nvidia/graph-composer/gxf_cli
update-alternatives --remove gxf_server /opt/nvidia/graph-composer/gxf_server
ldconfig
rm -rf /var/tmp/gxf
rm -rf /tmp/gxf_registry.log

%files
/opt/nvidia/graph-composer/*

%clean
rm -rf %{buildroot}

%changelog
* Thu Mar 21 2024 Jaiprkash Khemkaar <jrao@nvidia.com>
-
