@0x1bfebcdef79649c15;

struct Flow {
	id @0 :Data;
	inBytes @1 :UInt32;
	inPkts @2 :UInt32;
	protocol @3 :UInt8;
	tcpFlags @4 :UInt8;
	l4SrcPort @5 :UInt16;
	ipv4SrcAddr @6 :UInt32;
	ipv6SrcAddr @7 :Data;
	l4DstPort @8 :UInt16;
	ipv4DstAddr @9 :UInt32;
	ipv6DstAddr @10 :Data;
	outBytes @11 :UInt32;
	outPkts @12 :UInt32;
	minIpPktLen @13 :UInt16;
	maxIpPktLen @14 :UInt16;
	icmpType @15 :UInt16;
	minTtl @16 :UInt8;
	maxTtl @17 :UInt8;
	direction @18 :UInt8;
	flowStartMilliseconds @19 :UInt64;
	flowEndMilliseconds @20 :UInt64;
	srcFragments @21 :UInt16;
	dstFragments @22 :UInt16;
	clientTcpFlags @23 :UInt8;
	serverTcpFlags @24 :UInt8;
	srcToDstAvgThroughput @25 :UInt32;
	dstToSrcAvgThroughput @26 :UInt32;
	numPktsUpTo128Bytes @27 :UInt32;
	numPkts128To256Bytes @28 :UInt32;
	numPkts256To512Bytes @29 :UInt32;
	numPkts512To1024Bytes @30 :UInt32;
	numPkts1024To1514Bytes @31 :UInt32;
	numPktsOver1514Bytes @32 :UInt32;
	srcIpCountry @33 :Text;
	dstIpCountry @34 :Text;
	srcIpLong @35 :Text;
	srcIpLat @36 :Text;
	dstIpLong @37 :Text;
	dstIpLat @38 :Text;
	longestFlowPkt @39 :UInt16;
	shortestFlowPkt @40 :UInt16;
	retransmittedInPkts @41 :UInt32;
	retransmittedOutPkts @42 :UInt32;
	ooorderInPkts @43 :UInt32;
	ooorderOutPkts @44 :UInt32;
	durationIn @45 :UInt32;
	durationOut @46 :UInt32;
	tcpWinMinIn @47 :UInt16;
	tcpWinMaxIn @48 :UInt16;
	tcpWinMssIn @49 :UInt16;
	tcpWinScaleIn @50 :UInt8;
	tcpWinMinOut @51 :UInt16;
	tcpWinMaxOut @52 :UInt16;
	tcpWinMssOut @53 :UInt16;
	tcpWinScaleOut @54 :UInt8;
	flowVerdict @55 :UInt16;
	srcToDstIatMin @56 :UInt32;
	srcToDstIatMax @57 :UInt32;
	srcToDstIatAvg @58 :UInt32;
	srcToDstIatStddev @59 :UInt32;
	dstToSrcIatMin @60 :UInt32;
	dstToSrcIatMax @61 :UInt32;
	dstToSrcIatAvg @62 :UInt32;
	dstToSrcIatStddev @63 :UInt32;
	applicationId @64 :UInt32;
}