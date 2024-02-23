//
// Created by adamyuan on 2/23/24.
//

#include "Sobol.hpp"

#include <glm/glm.hpp>

const uint32_t kMatrices[64][32] = {
    // clang-format off
	{0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x8000000, 0x4000000, 0x2000000, 0x1000000, 0x800000, 0x400000, 0x200000, 0x100000, 0x80000, 0x40000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100, 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1, },
	{0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00, 0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff, },
	{0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000, 0x68800000, 0x9cc00000, 0xee600000, 0x55900000, 0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000, 0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000, 0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500, 0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590, 0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555, },
	{0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000, 0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000, 0x78080000, 0xb40c0000, 0x82020000, 0xc3050000, 0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000, 0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00, 0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050, 0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093, },
	{0x80000000, 0x40000000, 0x20000000, 0xb0000000, 0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000, 0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000, 0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000, 0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000, 0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00, 0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0, 0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d, },
	{0x80000000, 0x40000000, 0x60000000, 0x30000000, 0xc8000000, 0x24000000, 0x56000000, 0xfb000000, 0xe0800000, 0x70400000, 0xa8600000, 0x14300000, 0x9ec80000, 0xdf240000, 0xb6d60000, 0x8bbb0000, 0x48008000, 0x64004000, 0x36006000, 0xcb003000, 0x2880c800, 0x54402400, 0xfe605600, 0xef30fb00, 0x7e48e080, 0xaf647040, 0x1eb6a860, 0x9f8b1430, 0xd6c81ec8, 0xbb249f24, 0x80d6d6d6, 0x40bbbbbb, },
	{0x80000000, 0xc0000000, 0xa0000000, 0xd0000000, 0x58000000, 0x94000000, 0x3e000000, 0xe3000000, 0xbe800000, 0x23c00000, 0x1e200000, 0xf3100000, 0x46780000, 0x67840000, 0x78460000, 0x84670000, 0xc6788000, 0xa784c000, 0xd846a000, 0x5467d000, 0x9e78d800, 0x33845400, 0xe6469e00, 0xb7673300, 0x20f86680, 0x104477c0, 0xf8668020, 0x4477c010, 0x668020f8, 0x77c01044, 0x8020f866, 0xc0104477, },
	{0x80000000, 0x40000000, 0xa0000000, 0x50000000, 0x88000000, 0x24000000, 0x12000000, 0x2d000000, 0x76800000, 0x9e400000, 0x8200000, 0x64100000, 0xb2280000, 0x7d140000, 0xfea20000, 0xba490000, 0x1a248000, 0x491b4000, 0xc4b5a000, 0xe3739000, 0xf6800800, 0xde400400, 0xa8200a00, 0x34100500, 0x3a280880, 0x59140240, 0xeca20120, 0x974902d0, 0x6ca48768, 0xd75b49e4, 0xcc95a082, 0x87639641, },
	{0x80000000, 0x40000000, 0xa0000000, 0x50000000, 0x28000000, 0xd4000000, 0x6a000000, 0x71000000, 0x38800000, 0x58400000, 0xea200000, 0x31100000, 0x98a80000, 0x8540000, 0xc22a0000, 0xe5250000, 0xf2b28000, 0x79484000, 0xfaa42000, 0xbd731000, 0x18a80800, 0x48540400, 0x622a0a00, 0xb5250500, 0xdab28280, 0xad484d40, 0x90a426a0, 0xcc731710, 0x20280b88, 0x10140184, 0x880a04a2, 0x84350611, },
	{0x80000000, 0x40000000, 0xe0000000, 0xb0000000, 0x98000000, 0x94000000, 0x8a000000, 0x5b000000, 0x33800000, 0xd9c00000, 0x72200000, 0x3f100000, 0xc1b80000, 0xa6ec0000, 0x53860000, 0x29f50000, 0xa3a8000, 0x1b2ac000, 0xd392e000, 0x69ff7000, 0xea380800, 0xab2c0400, 0x4ba60e00, 0xfde50b00, 0x60028980, 0xf006c940, 0x7834e8a0, 0x241a75b0, 0x123a8b38, 0xcf2ac99c, 0xb992e922, 0x82ff78f1, },
	{0x80000000, 0x40000000, 0xa0000000, 0x10000000, 0x8000000, 0x6c000000, 0x9e000000, 0x23000000, 0x57800000, 0xadc00000, 0x7fa00000, 0x91d00000, 0x49880000, 0xced40000, 0x880a0000, 0x2c0f0000, 0x3e0d8000, 0x3317c000, 0x5fb06000, 0xc1f8b000, 0xe18d8800, 0xb2d7c400, 0x1e106a00, 0x6328b100, 0xf7858880, 0xbdc3c2c0, 0x77ba63e0, 0xfdf7b330, 0xd7800df8, 0xedc0081c, 0xdfa0041a, 0x81d00a2d, },
	{0x80000000, 0x40000000, 0x20000000, 0x30000000, 0x58000000, 0xac000000, 0x96000000, 0x2b000000, 0xd4800000, 0x9400000, 0xe2a00000, 0x52500000, 0x4e280000, 0xc71c0000, 0x629e0000, 0x12670000, 0x6e138000, 0xf731c000, 0x3a98a000, 0xbe449000, 0xf83b8800, 0xdc2dc400, 0xee06a200, 0xb7239300, 0x1aa80d80, 0x8e5c0ec0, 0xa03e0b60, 0x703701b0, 0x783b88c8, 0x9c2dca54, 0xce06a74a, 0x87239795, },
	{0x80000000, 0xc0000000, 0xa0000000, 0x50000000, 0xf8000000, 0x8c000000, 0xe2000000, 0x33000000, 0xf800000, 0x21400000, 0x95a00000, 0x5e700000, 0xd8080000, 0x1c240000, 0xba160000, 0xef370000, 0x15868000, 0x9e6fc000, 0x781b6000, 0x4c349000, 0x420e8800, 0x630bcc00, 0xf7ad6a00, 0xad739500, 0x77800780, 0x6d4004c0, 0xd7a00420, 0x3d700630, 0x2f880f78, 0xb1640ad4, 0xcdb6077a, 0x824706d7, },
	{0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0x38000000, 0xc4000000, 0x42000000, 0xa3000000, 0xf1800000, 0xaa400000, 0xfce00000, 0x85100000, 0xe0080000, 0x500c0000, 0x58060000, 0x54090000, 0x7a038000, 0x670c4000, 0xb3842000, 0x94a3000, 0xd6f1800, 0x2f5aa400, 0x1ce7ce00, 0xd5145100, 0xb8000080, 0x40000c0, 0x22000060, 0x33000090, 0xc9800038, 0x6e4000c4, 0xbee00042, 0x261000a3, },
	{0x80000000, 0x40000000, 0x20000000, 0xf0000000, 0xa8000000, 0x54000000, 0x9a000000, 0x9d000000, 0x1e800000, 0x5cc00000, 0x7d200000, 0x8d100000, 0x24880000, 0x71c40000, 0xeba20000, 0x75df0000, 0x6ba28000, 0x35d14000, 0x4ba3a000, 0xc5d2d000, 0xe3a16800, 0x91db8c00, 0x79aef200, 0xcdf4100, 0x672a8080, 0x50154040, 0x1a01a020, 0xdd0dd0f0, 0x3e83e8a8, 0xaccacc54, 0xd52d529a, 0xd91d919d, },
	{0x80000000, 0xc0000000, 0x20000000, 0xd0000000, 0xd8000000, 0xc4000000, 0x46000000, 0x85000000, 0xa5800000, 0x76c00000, 0xada00000, 0x6ab00000, 0x2da80000, 0xaabc0000, 0xdaa0000, 0x7ab10000, 0xd5a78000, 0xbebd4000, 0x93a3e000, 0x3bb51000, 0x3629b800, 0x4d727c00, 0x9b836200, 0x27c4d700, 0xb629b880, 0x8d727cc0, 0xbb836220, 0xf7c4d7d0, 0x6e29b858, 0x49727c04, 0xfd836266, 0x72c4d755, },
	{0x80000000, 0x40000000, 0x20000000, 0xf0000000, 0x38000000, 0x14000000, 0xf6000000, 0x67000000, 0x8f800000, 0x50400000, 0x8aa00000, 0xff00000, 0x12a80000, 0xabf40000, 0xfcaa0000, 0x28fb0000, 0xbd298000, 0xbba4000, 0x4e06e000, 0x330c3000, 0x59861800, 0xc74d3400, 0x3d2cb200, 0x4bb2cb00, 0x6e061880, 0xc30d3440, 0x618cb220, 0xd342cbf0, 0xcb2e18b8, 0x2cb93454, 0xe186b2d6, 0x9349cb97, },
	{0x80000000, 0xc0000000, 0x20000000, 0xf0000000, 0x68000000, 0x64000000, 0x36000000, 0x6d000000, 0x41800000, 0xe0400000, 0xd2e00000, 0x9bf00000, 0xce80000, 0x52fc0000, 0x5b6a0000, 0x2fb30000, 0xa00c8000, 0x30054000, 0x4807e000, 0x940f9000, 0x5e01f800, 0x90e9400, 0x778a5600, 0x8d416b00, 0x9369f880, 0x7bb294c0, 0xde005620, 0xc9026bf0, 0x578d78e8, 0x7d4bd4a4, 0xfb6db616, 0x1fbefb9d, },
	{0x80000000, 0x40000000, 0xa0000000, 0x50000000, 0x98000000, 0xf4000000, 0xae000000, 0xbb000000, 0xe7800000, 0x95c00000, 0x1c200000, 0xd0300000, 0xdba80000, 0x55f40000, 0xff820000, 0x21c10000, 0x12238000, 0x3b3a4000, 0xa42b6000, 0x3430f000, 0x4da69800, 0x4af3ec00, 0x2e043a00, 0xfb0a1f00, 0x47851880, 0xc5c9ac40, 0x842f5aa0, 0x243aef50, 0x75a38018, 0xeefa40b4, 0x180b600e, 0xb400f0eb, },
	{0x80000000, 0xc0000000, 0xe0000000, 0xb0000000, 0xb8000000, 0x3c000000, 0xce000000, 0x41000000, 0x21800000, 0x51c00000, 0x9600000, 0x85700000, 0xf2780000, 0x8e9c0000, 0x60020000, 0x70030000, 0x58038000, 0x8c02c000, 0x7602e000, 0x7d00f000, 0xef833800, 0x10c10400, 0x28e08600, 0xd4b14700, 0xfb182580, 0xbee15c0, 0x9279c9e0, 0xfe9d3a70, 0x38000008, 0xfc00000c, 0x2e00000e, 0xf100000b, },
	{0x80000000, 0xc0000000, 0xe0000000, 0xd0000000, 0x68000000, 0x3c000000, 0x8a000000, 0x51000000, 0xa9800000, 0xddc00000, 0x5ba00000, 0x39d00000, 0x95f80000, 0x56d40000, 0xa020000, 0x91030000, 0x49838000, 0xdc34000, 0x33a1a000, 0x5d0f000, 0x1ffa2800, 0x7d54400, 0xa380a600, 0x4cc07700, 0x1222ee80, 0x3413a740, 0xa65bf7e0, 0x5305ab50, 0x15f80008, 0x96d4000c, 0xea02000e, 0x4103000d, },
	{0x80000000, 0x40000000, 0x60000000, 0xd0000000, 0x38000000, 0x8c000000, 0x7e000000, 0x71000000, 0xc8800000, 0x4c00000, 0x1ba00000, 0xbb700000, 0x4a980000, 0xc3bc0000, 0xa6020000, 0x6d010000, 0xee818000, 0x29c34000, 0x9520e000, 0x42b23000, 0xe7b9f800, 0xd0dc400, 0x3fb92200, 0x110d1300, 0x19bbee80, 0x3c0cadc0, 0x973a4a60, 0xc5cf7ef0, 0x3a180008, 0xb7c0004, 0xa3a20006, 0x7771000d, },
	{0x80000000, 0xc0000000, 0xa0000000, 0x90000000, 0x8000000, 0x64000000, 0x6a000000, 0x89000000, 0xa5800000, 0xcb400000, 0x18200000, 0xad900000, 0xaf880000, 0x72f40000, 0x25820000, 0xb430000, 0xb8228000, 0x3d924000, 0xa7882000, 0x16f59000, 0x4f83a800, 0x82412400, 0x1da01600, 0xf6d16d00, 0xbfa84080, 0xbb672640, 0xe0091620, 0xf0b4efd0, 0x38228008, 0xfd92400c, 0x788200a, 0x86f59009, },
	{0x80000000, 0xc0000000, 0x20000000, 0xd0000000, 0x48000000, 0x8c000000, 0xd6000000, 0x39000000, 0xd5800000, 0x32400000, 0xb2a00000, 0x72100000, 0x53d80000, 0x82cc0000, 0xcb820000, 0x47430000, 0x91208000, 0xa9534000, 0x7cf92000, 0x4e9e3000, 0xfcf95800, 0x8e9fe400, 0xdcf9d600, 0x5e9c8900, 0x94f96a80, 0xd29fb840, 0x42f9b760, 0xeb9c9f30, 0x97788008, 0xd9df400c, 0x25db2002, 0xabcd300d, },
	{0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xd8000000, 0xf4000000, 0x3e000000, 0x95000000, 0x8f800000, 0x3d400000, 0xf3200000, 0x2ef00000, 0xadc80000, 0xa0c0000, 0x8b220000, 0x4af30000, 0x6bc88000, 0x3b0d4000, 0xe2a16000, 0x16b0d000, 0x29687800, 0xbdbf1400, 0x33cb5e00, 0xf0c2500, 0xfca1b480, 0xd3b0afc0, 0x7eeb6920, 0x74fe4d30, 0xfee87808, 0xb4ff140c, 0xdeeb5e02, 0xe4fc2505, },
	{0x80000000, 0x40000000, 0xa0000000, 0xb0000000, 0x98000000, 0xa4000000, 0x7a000000, 0xd5000000, 0x2800000, 0x60400000, 0x51e00000, 0x88700000, 0x8c280000, 0x47c40000, 0xbe20000, 0xad710000, 0xb6aa8000, 0x3386c000, 0xb8006000, 0x54039000, 0x42036800, 0xc1019400, 0xe0826a00, 0x11431100, 0x2960af80, 0x3d3175c0, 0xdf4a3aa0, 0xaff49e10, 0xd62b6808, 0x62c59404, 0x31606a0a, 0xd932110b, },
	{0x80000000, 0xc0000000, 0xa0000000, 0x30000000, 0x18000000, 0x34000000, 0x8a000000, 0x9d000000, 0x67800000, 0x82400000, 0x40e00000, 0x60f00000, 0x91480000, 0x29440000, 0x2d620000, 0xbfb30000, 0x162a8000, 0xfbf4c000, 0xe4ca6000, 0xc207d000, 0x2002a800, 0xf001b400, 0xb8037e00, 0x4021900, 0x92034b80, 0xa90327c0, 0xed81f320, 0x1f40d810, 0x27602808, 0xe2b1740c, 0xd1ab1e0a, 0x49b6c903, },
	{0x80000000, 0x40000000, 0xe0000000, 0xd0000000, 0x8000000, 0x4c000000, 0x2000000, 0xb5000000, 0x36800000, 0xc2c00000, 0x14200000, 0x7500000, 0x1bf80000, 0x50340000, 0x48a20000, 0xac910000, 0xd35b8000, 0xbca74000, 0x7bfa2000, 0xc0343000, 0xa0a18800, 0x30909400, 0xd95b7a00, 0x45a57b00, 0x4f7a7880, 0xb7f6f940, 0x82013de0, 0xf502dfd0, 0xd6820808, 0x12c3d404, 0x1c235a0e, 0x4b504b0d, },
	{0x80000000, 0xc0000000, 0xe0000000, 0x50000000, 0x68000000, 0x4c000000, 0x76000000, 0xf7000000, 0x36800000, 0xd7400000, 0x87e00000, 0xef300000, 0xa3a80000, 0xd5440000, 0x23aa0000, 0x15470000, 0xc3a98000, 0x45464000, 0xaba82000, 0x9477000, 0xdda9f800, 0xfe44ac00, 0xeb292200, 0x2907f100, 0x6ccb3d80, 0xc6344dc0, 0xcf61b320, 0x137318d0, 0xeccb3d88, 0x6344dcc, 0x2f61b32e, 0x437318d5, },
	{0x80000000, 0x40000000, 0x60000000, 0x90000000, 0xc8000000, 0x74000000, 0x52000000, 0x3000000, 0xeb800000, 0x6f400000, 0x64600000, 0xdaf00000, 0x17980000, 0x297c0000, 0xa59a0000, 0xfa7d0000, 0xe61b8000, 0x713f4000, 0x1878a000, 0xdcce9000, 0xb661e800, 0x99f29c00, 0x9c184600, 0xd63e2100, 0x9fa5780, 0x548e0ac0, 0xa380a9e0, 0x5b413f30, 0x56625788, 0x49f20ac4, 0x341aa9e6, 0x323c3f39, },
	{0x80000000, 0xc0000000, 0xa0000000, 0xd0000000, 0xb8000000, 0x4000000, 0x6e000000, 0x97000000, 0xf2800000, 0xedc00000, 0x13600000, 0x5c900000, 0xdb580000, 0x31e40000, 0x9da0000, 0xcc270000, 0x2b88000, 0x44b44000, 0xfe26000, 0xe6505000, 0x9ab9d800, 0x50b50c00, 0x79e29200, 0xa552fb00, 0xbe38bf80, 0x2e77d940, 0xf6000ae0, 0x830112d0, 0x84803f88, 0xaec3994c, 0x37e26aea, 0x225142dd, },
	{0x80000000, 0xc0000000, 0xe0000000, 0x30000000, 0x68000000, 0xec000000, 0x22000000, 0x2b000000, 0x36800000, 0x9d400000, 0x6a200000, 0x16700000, 0x4de80000, 0x330c0000, 0x936a0000, 0x824f0000, 0x3b498000, 0x8f3fc000, 0x28202000, 0xcd707000, 0xf36aa800, 0x724fdc00, 0xb34bf200, 0x533e6900, 0x62207a80, 0xa7140c0, 0xe7ea6520, 0xc40d90f0, 0xefe9fa88, 0xd80e80cc, 0x45ea452e, 0x2f0de0f3, },
	{0x80000000, 0xc0000000, 0x20000000, 0x30000000, 0x28000000, 0xd4000000, 0x8a000000, 0xff000000, 0x84800000, 0x73c00000, 0x13200000, 0xc2b00000, 0xfb380000, 0x361c0000, 0x401a0000, 0xe0af0000, 0x11228000, 0x19b3c000, 0xfdb82000, 0x5edf9000, 0x75b88800, 0x7adfac00, 0xf7baba00, 0x61ddf300, 0xd1387e80, 0x391e55c0, 0xcc9ba860, 0x776cbeb0, 0xa000f688, 0xf001f9cc, 0x8011262, 0xe4014db3, },
	{0x80000000, 0x40000000, 0xa0000000, 0x50000000, 0xb8000000, 0x84000000, 0x1a000000, 0xaf000000, 0xbd800000, 0xdfc00000, 0x14e00000, 0x43500000, 0xda380000, 0x4e1c0000, 0x4cda0000, 0x364d0000, 0x29608000, 0xdc904000, 0x6ed86000, 0x5d4f5000, 0x2ee08800, 0xfc51ac00, 0x7fb81e00, 0x45dc8300, 0xfa3a4580, 0x5e1d6240, 0x54dbd360, 0xe24ec930, 0x8b62cd88, 0xf790ce44, 0xc959cd6a, 0x2d8f4a35, },
	{0x80000000, 0x40000000, 0xe0000000, 0x70000000, 0x8000000, 0xf4000000, 0xf6000000, 0x8b000000, 0xc9800000, 0x55400000, 0x67200000, 0xf3f00000, 0x34780000, 0x57440000, 0x1ada0000, 0xb1f50000, 0xa9818000, 0x6540c000, 0x8f23a000, 0x77f21000, 0xca7bf800, 0x2845fc00, 0x255afe00, 0x6fb67900, 0x7233a80, 0xc3f25ac0, 0xdc7aed60, 0xd34482d0, 0xe4d94288, 0xcef766c4, 0x9603b36e, 0xbb00ebd7, },
	{0x80000000, 0x40000000, 0xe0000000, 0x90000000, 0x68000000, 0xf4000000, 0x62000000, 0xdf000000, 0x79800000, 0xdd400000, 0x76e00000, 0x2cf00000, 0xcfb80000, 0x51ec0000, 0xc8da0000, 0x845d0000, 0x9b818000, 0x42434000, 0xef622000, 0x61b19000, 0xd1582800, 0x891cac00, 0x65626e00, 0xab10900, 0x2adbbd80, 0x1b5d86c0, 0x2014560, 0xf032470, 0xf1821588, 0xb9426ac4, 0x7ce10b6e, 0x7f3bd79, },
	{0x80000000, 0xc0000000, 0x60000000, 0x50000000, 0x18000000, 0xdc000000, 0x42000000, 0x37000000, 0x20800000, 0xf1400000, 0x28600000, 0x94900000, 0x87880000, 0xa83c0000, 0x556a0000, 0xe6ef0000, 0xf8038000, 0x4c024000, 0x3a01e000, 0xbb023000, 0x7a816800, 0x1a43ac00, 0x4ae18a00, 0x52d31900, 0x8f682380, 0xcded9740, 0xfa80bfa0, 0xda43f2b0, 0x2ae2cb88, 0x2d07b4c, 0x976ad5a6, 0x11eddbb5, },
	{0x80000000, 0xc0000000, 0x20000000, 0xf0000000, 0xf8000000, 0x34000000, 0x62000000, 0xf5000000, 0xa8800000, 0xfcc00000, 0x8e200000, 0x53f00000, 0xc7780000, 0x95740000, 0xb8020000, 0xd4e50000, 0xb2808000, 0xfdc0c000, 0x64a02000, 0xaa30f000, 0x19d8f800, 0xe443400, 0x935a6200, 0xe761f500, 0x657a2880, 0x40913cc0, 0xe0022e20, 0xd0e563f0, 0x8809f78, 0xccc09174, 0x56200202, 0x97f0e5e5, },
	{0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0xf8000000, 0xec000000, 0x7e000000, 0x61000000, 0x5c800000, 0xe6c00000, 0xdda00000, 0x2a700000, 0x93380000, 0x13cc0000, 0xd3ce0000, 0x73790000, 0x83a08000, 0x7b70c000, 0x97b8a000, 0xe90cf000, 0x886ef800, 0xd409ec00, 0x3218fe00, 0xef7ca100, 0xc556fc80, 0x56c516c0, 0x4556a5a0, 0x96c50670, 0xe556cd38, 0x66c542cc, 0x1d56574e, 0x8ac549b9, },
	{0x80000000, 0xc0000000, 0x20000000, 0xb0000000, 0x58000000, 0x2c000000, 0x9a000000, 0xf9000000, 0x3c800000, 0xb2c00000, 0xad200000, 0x3a300000, 0x89980000, 0x448c0000, 0x2eea0000, 0x6f810000, 0xef208000, 0x2f30c000, 0xf182000, 0xbf4cb000, 0xe74a5800, 0xcb712c00, 0x51981a00, 0xa88c3900, 0x94ea1c80, 0x268102c0, 0x8ba07520, 0xb1f0d630, 0x38383398, 0x7c7c0d8c, 0x52524a6a, 0x3d3df141, },
	{0x80000000, 0xc0000000, 0x20000000, 0xb0000000, 0xd8000000, 0xac000000, 0x8e000000, 0x9000000, 0x9e800000, 0xa1c00000, 0xcaa00000, 0x33700000, 0x95780000, 0x85c0000, 0x24b60000, 0x6a350000, 0x43788000, 0x6d5cc000, 0x14362000, 0x72f5b000, 0xcf585800, 0x53ec6c00, 0xc5eeae00, 0x40d9b900, 0xe016c680, 0x9045cdc0, 0x6880e4a0, 0x74c04a70, 0x2220f3f8, 0x87b0b59c, 0x9758b816, 0x3fecfc45, },
	{0x80000000, 0x40000000, 0xe0000000, 0xf0000000, 0xa8000000, 0x2c000000, 0xa2000000, 0x2d000000, 0xda800000, 0xf9400000, 0xec600000, 0x2b00000, 0x3d480000, 0x825c0000, 0x7d4a0000, 0x62610000, 0x8dc88000, 0xca1c4000, 0xa1aae000, 0x6891f000, 0x8c602800, 0xb2b06c00, 0x75484200, 0x5e5cdd00, 0x774a7280, 0x6361d540, 0xf548ce60, 0x1e5c6fb0, 0x974a07c8, 0x93618b1c, 0x5d48b92a, 0x325c0cd1, },
	{0x80000000, 0xc0000000, 0xe0000000, 0x30000000, 0xc8000000, 0x7c000000, 0x82000000, 0x4f000000, 0xbe800000, 0xedc00000, 0x21600000, 0xab700000, 0x78680000, 0x746c0000, 0x1e9a0000, 0xfdcb0000, 0x39088000, 0x2f1cc000, 0x4ef2e000, 0xc5a73000, 0x6d924800, 0xe1d7bc00, 0x4b7ae200, 0x487bbf00, 0xbc801680, 0x62c061c0, 0x7fe08b60, 0x76b0a870, 0x91088ce8, 0xa31caaac, 0xe4f2037a, 0xc6a7f47b, },
	{0x80000000, 0xc0000000, 0x20000000, 0x10000000, 0x98000000, 0x2c000000, 0x6000000, 0xcd000000, 0x8a800000, 0x1bc00000, 0xffa00000, 0xad500000, 0x7af80000, 0xb3dc0000, 0x5b2e0000, 0x1f290000, 0x9d588000, 0xf28cc000, 0x7d62000, 0x71f51000, 0xd4f61800, 0xda65ec00, 0x632ea600, 0xe3291d00, 0x2358b280, 0x38ce7c0, 0x135641a0, 0x8b355c50, 0xa7d6ee78, 0xa1f5891c, 0x6cf6880e, 0xe665b4b9, },
	{0x80000000, 0x40000000, 0xa0000000, 0x90000000, 0x98000000, 0x54000000, 0x3a000000, 0x9d000000, 0x7e800000, 0x7f400000, 0x17200000, 0xab500000, 0x6df80000, 0x96a40000, 0x83d20000, 0x71e10000, 0xc0d88000, 0xe0f44000, 0x30aaa000, 0x8059000, 0xcc2a1800, 0x6e451400, 0xa78a1a00, 0xe3554d00, 0x1d2c680, 0x68e1fb40, 0xbc589520, 0xc6b4b250, 0xfb0a1178, 0x1515b0e4, 0xf272c872, 0xb1f12cf1, },
	{0x80000000, 0xc0000000, 0xe0000000, 0xb0000000, 0x8000000, 0x84000000, 0xb2000000, 0xb9000000, 0xbe800000, 0x4fc00000, 0x55600000, 0xf8f00000, 0xac280000, 0x66d40000, 0xb30a0000, 0x8bb50000, 0xc7c88000, 0x11e4c000, 0xaa42e000, 0xa591b000, 0xd0ea8800, 0x78854400, 0x6c80d200, 0x86c0c900, 0x3e05680, 0x83307bc0, 0x4348ef60, 0xa324c5f0, 0x13a2a0a8, 0x1ba19014, 0x9f22d8ea, 0x2d61fc85, },
	{0x80000000, 0xc0000000, 0x60000000, 0x30000000, 0x78000000, 0x24000000, 0x9e000000, 0x47000000, 0x67800000, 0xf7400000, 0xdf200000, 0xb3100000, 0x71680000, 0x8c4c0000, 0x32520000, 0xe5d50000, 0xaa528000, 0x31d5c000, 0x2c52e000, 0x62d5f000, 0xadd29800, 0xf695d400, 0x8b720600, 0xf5c59300, 0x42ba6180, 0x3dd96440, 0xdea0bea0, 0xe750d750, 0x37c84fc8, 0xbf1c9b1c, 0x839a1d9a, 0x9c94ec9, },
	{0x80000000, 0xc0000000, 0xe0000000, 0xb0000000, 0x78000000, 0x9c000000, 0xee000000, 0x1b000000, 0xcb800000, 0xc3400000, 0xc7a00000, 0x5100000, 0x88680000, 0xc4740000, 0x225a0000, 0x3da10000, 0x345a8000, 0x7aa1c000, 0xf1da6000, 0x12e17000, 0x85fa1800, 0x48b1ec00, 0x2432f600, 0x92d5f700, 0x45803d80, 0xa8403440, 0x94207a20, 0xea50f150, 0xd9c81248, 0x46648524, 0x8fb24812, 0x21952485, },
	{0x80000000, 0x40000000, 0x60000000, 0x10000000, 0x58000000, 0x7c000000, 0xc2000000, 0xe1000000, 0xd800000, 0xd7c00000, 0x2aa00000, 0xf5300000, 0x9ba80000, 0xc0f40000, 0x20c60000, 0x702f0000, 0x48668000, 0x241f4000, 0xbe4ee000, 0x232b5000, 0xec28b800, 0xda342c00, 0xfde6fa00, 0xdfdf8d00, 0x6eee1780, 0x5b1b0ac0, 0xe0000520, 0x500093f0, 0x38008488, 0x6c008e04, 0x9a000bce, 0x9d00d8eb, },
	{0x80000000, 0x40000000, 0x20000000, 0x30000000, 0xb8000000, 0xac000000, 0x72000000, 0xb1000000, 0x3800000, 0xd2c00000, 0xc1600000, 0x9b900000, 0x4e480000, 0xb740000, 0x864e0000, 0x3f0b0000, 0x68068000, 0x447f4000, 0x7648a000, 0xe7747000, 0xd44e9800, 0xbe0b9c00, 0xd3864a00, 0x3abf5d00, 0xc528d180, 0xcde413c0, 0x99865ae0, 0x67bfd550, 0x94a8c528, 0x9e24cde4, 0xe3669986, 0x82ef67bf, },
	{0x80000000, 0xc0000000, 0xe0000000, 0x70000000, 0x88000000, 0x44000000, 0x4a000000, 0x47000000, 0xdd800000, 0x42400000, 0xc3200000, 0x77100000, 0x75b80000, 0x966c0000, 0x715e0000, 0xfc950000, 0xa6e68000, 0xd9f9c000, 0x28386000, 0x142cb000, 0x527e6800, 0xfb853400, 0x5b5e4200, 0xb95c300, 0x1366f780, 0xafb9b540, 0x2918f6a0, 0x603cc150, 0xb0469498, 0x68a9927c, 0x34a09b66, 0xc250ebb9, },
	{0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xd8000000, 0xfc000000, 0xf6000000, 0xd5000000, 0xbf800000, 0x2c400000, 0xeee00000, 0x9700000, 0x19080000, 0x21640000, 0xad6a0000, 0xd3130000, 0x22828000, 0x9707c000, 0x98e0a000, 0x1c709000, 0x8688f800, 0x5d24ac00, 0x9b8a2e00, 0x26632900, 0xcd8ac980, 0x63633940, 0x8a0af160, 0xe323b530, 0x4aea8fe8, 0xc3534414, 0x1a623a62, 0x1b774b77, },
	{0x80000000, 0x40000000, 0x60000000, 0x50000000, 0x58000000, 0xac000000, 0x6a000000, 0x85000000, 0xfb800000, 0xa8c00000, 0x84200000, 0xae300000, 0x4b080000, 0xe0740000, 0x10860000, 0x388f0000, 0xfc2e8000, 0x320b4000, 0x2980e000, 0x91c01000, 0x2da03800, 0x7ff0fc00, 0x6a83200, 0xcf842900, 0x4e2e9180, 0x5b0b2dc0, 0xd800ffa0, 0xec0046f0, 0xa00af28, 0xd5001e44, 0xa380038e, 0x4c074fb, },
	{0x80000000, 0xc0000000, 0xa0000000, 0x50000000, 0xe8000000, 0x44000000, 0x5e000000, 0xad000000, 0xef800000, 0x68400000, 0x84600000, 0xfe500000, 0xfd280000, 0x7f40000, 0x2c620000, 0xda4f0000, 0x53068000, 0x12dfc000, 0x6f802000, 0xa8403000, 0x24602800, 0xae501400, 0x15283a00, 0x43f41100, 0x72621780, 0x774f2b40, 0xbc86bbe0, 0x7a9fda10, 0xebe00118, 0x56100f94, 0xd948174a, 0xa9a415fd, },
	{0x80000000, 0xc0000000, 0x60000000, 0xb0000000, 0x18000000, 0x4000000, 0xda000000, 0x9000000, 0x22800000, 0xe8400000, 0xbc600000, 0xe300000, 0x7b580000, 0x378c0000, 0x14c20000, 0x874d0000, 0x99d48000, 0xbfb94000, 0x18802000, 0x91403000, 0xe6e01800, 0x52702c00, 0x5380600, 0x34bc0100, 0x971a3680, 0x51810240, 0x13f688a0, 0xde847a10, 0x466c8f18, 0x1745738c, 0x91fa26d6, 0x73f111e3, },
	{0x80000000, 0x40000000, 0x20000000, 0x50000000, 0x88000000, 0x9c000000, 0x2e000000, 0x5000000, 0xab800000, 0x1c400000, 0x6e200000, 0x25100000, 0xfba80000, 0x94040000, 0xf26e0000, 0xb070000, 0xfeaa8000, 0x3fd1c000, 0xee202000, 0x65101000, 0xdba80800, 0xc4041400, 0x7a6e2200, 0x97072700, 0xd0aa8b80, 0x3ad1c140, 0x45a00ae0, 0x79501710, 0xb5881388, 0xe1141d44, 0x81c61cea, 0x3030201, },
	{0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xc8000000, 0x3c000000, 0x3e000000, 0x67000000, 0xf9800000, 0xcc400000, 0x66600000, 0xb3100000, 0xaba80000, 0x5d240000, 0xc4fe0000, 0xb8cf0000, 0x66bb8000, 0x71a8c000, 0x10602000, 0x28103000, 0x4c280800, 0xa6641400, 0x931e3200, 0xfb9f0f00, 0x95738f80, 0xf89cd9c0, 0x86b61e60, 0x1bb0310, 0x880d9198, 0xdc13f8c4, 0x4e6db8ea, 0xff03e849, },
	{0x80000000, 0x40000000, 0x20000000, 0xb0000000, 0x58000000, 0x44000000, 0x7e000000, 0x69000000, 0x5b800000, 0xdc400000, 0x5a200000, 0x87100000, 0xdad80000, 0x9bec0000, 0xbc420000, 0xca0f0000, 0x6f7c8000, 0xc6d9c000, 0xa1a02000, 0xab501000, 0xf8f80800, 0xe8fc2c00, 0x409a1600, 0x7ce31100, 0xf6be9f80, 0xb996da40, 0xcf7cb6e0, 0x36d9e710, 0xd9a03e88, 0x5f501dc4, 0xdef828b6, 0xc5fc1bfb, },
	{0x80000000, 0x40000000, 0xa0000000, 0xb0000000, 0x48000000, 0x74000000, 0xc2000000, 0xe7000000, 0xb5800000, 0xba400000, 0x9b200000, 0xa3d00000, 0x2f180000, 0x81840000, 0xd82a0000, 0xcc190000, 0x5e078000, 0xe138c000, 0xd8982000, 0x9cc41000, 0x568a2800, 0x65892c00, 0xa23f9200, 0xb76cdd00, 0xedaa1080, 0x365929c0, 0x65278560, 0xf2e8c290, 0xbf8014c8, 0x694025f4, 0x4ca01346, 0x4e9035a1, },
	{0x80000000, 0x40000000, 0xa0000000, 0xf0000000, 0x98000000, 0xb4000000, 0x52000000, 0x7000000, 0xbf800000, 0x5a400000, 0x3b200000, 0x91d00000, 0xd3380000, 0xfdec0000, 0x954a0000, 0x58f10000, 0xb5df8000, 0x91dc000, 0x86b82000, 0xa4ac1000, 0x7bea2800, 0xd0613c00, 0x2847a600, 0x8c61ed00, 0x166a3480, 0xcd2111c0, 0xce787e0, 0xb7f1ea90, 0x667208c8, 0x151d1974, 0x1895884e, 0x15ecc2bb, },
	{0x80000000, 0xc0000000, 0xe0000000, 0x70000000, 0xf8000000, 0x4c000000, 0xa6000000, 0x89000000, 0x6e800000, 0x1a400000, 0x17600000, 0x4bf00000, 0xa2f80000, 0x7c5c0000, 0x7e360000, 0x551b0000, 0x40808000, 0x272d4000, 0x93982000, 0x7eac3000, 0x524e3800, 0x43071c00, 0xd1d6be00, 0x75c65300, 0xd7e08980, 0xacdd5240, 0xd16003a0, 0x72f02a90, 0xd47803d8, 0x5a1c1dfc, 0x37563f3e, 0xdbeb2e57, },
	{0x80000000, 0x40000000, 0x20000000, 0x30000000, 0xb8000000, 0x3c000000, 0xde000000, 0xdf000000, 0x29800000, 0x32400000, 0xe9200000, 0x62900000, 0x71d80000, 0x5e3c0000, 0x9f2e0000, 0x9e70000, 0x26b8000, 0x5176c000, 0x5ef82000, 0xafac1000, 0x81760800, 0xb69b0c00, 0x3be5ae00, 0xeb41cf00, 0x33eb9780, 0x2f36e7c0, 0xf1d82260, 0x1e3c1090, 0xbf2e1c48, 0x39e71ba4, 0xba6b85f6, 0x6d76ef4f, },
	{0x80000000, 0x40000000, 0xa0000000, 0xd0000000, 0xf8000000, 0x3c000000, 0x6e000000, 0x19000000, 0x50800000, 0xca400000, 0x7b200000, 0xafd00000, 0x97a80000, 0x4b9c0000, 0x55ae0000, 0x64ef0000, 0xf0288000, 0x68524000, 0x64082000, 0x820c1000, 0x8f262800, 0x75a33400, 0xf4aebe00, 0xa8614f00, 0x842ebb80, 0xf2215640, 0xa70e9c20, 0xb1f15690, 0xa6a6a8c8, 0xdf6d40f4, 0xcd88886a, 0x68c27fa7, },
	{0x80000000, 0x40000000, 0x60000000, 0xd0000000, 0xc8000000, 0xbc000000, 0x4e000000, 0x57000000, 0x80800000, 0xa400000, 0xfd200000, 0x8db00000, 0xffa80000, 0xa6840000, 0x110e0000, 0x4bdf0000, 0x74d78000, 0xb8724000, 0x84082000, 0x8a741000, 0xbd061800, 0xedab3400, 0x2fd1b200, 0x6ed96f00, 0xad59b380, 0x5ed45c0, 0x23ff9820, 0x38b66690, 0x8e263548, 0x771b286c, 0x30f9866a, 0x121d6761, },
    // clang-format on
};

void Sobol::Next() {
	if (m_index == -1u)
		m_index = 0u;
	uint32_t c = glm::findLSB(~(m_index++));
	for (uint32_t i = 0; i < kDimension; ++i)
		m_sequence[i] ^= kMatrices[i][c];
}