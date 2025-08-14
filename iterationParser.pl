# my $inputFile = "vllm_server_log_scheinfo.log";

# my $model = "qwen2.5-32b";
#my $model = "llama3-70b";
my $inputFile = $ARGV[0];
my $model = $ARGV[1];
my $numGPUs = $ARGV[2];
my $weightBits = $ARGV[3];
print "inputFile: $inputFile\n";
print "model: $model\n";

my $cmdPrefix = "python ../llama3/main.py 1 -m $model -d bmg24 -g $numGPUs -w $weightBits -v \"0.3 0.8 0.7 0.85\" ";
# my $cmdPrefix = "python ../llama3/main.py 1 -m $model -d bmg24 -g $numGPUs -w $weightBits ";

my $iteration = 0;
my @projetedTimeData = ();
my @actualTimeData = ();
my @contextData = ();
my @tokenData = ();
my @cmdPrefixData = ();

sub findNext;

open FIN, $inputFile;
while (1)
{
    my $requests = findNext "total_requests_num:\\s*\\d+";
    if (!defined($requests))
    {
        last;
    }
    $requests =~ m/\d+$/;
    $requests = $&;
    #print "Request: $requests\n";
    if ($requests > 0)
    {
        my $end = 0;
        my @lengths = ();
        my @contexts = ();        
        for (my $i = 0; $i < $requests; $i++)
        {
            my $requestID = findNext "request id:\\s*.+\$";
            if (!defined($requestID))
            {
                $end = 1;
                last;
            }
            $requestID =~ s/request id:\s*//g;
            #print "RequestID: $requestID\n";
            my $context = findNext "context_len:\\s*\\d+";
            if (!defined($context))
            {
                $end = 1;
                last;
            }
            $context =~ m/\d+$/;
            $context = $&;
            my $length = findNext "num_scheduled_token:\\s*\\d+";
            if (!defined($length))
            {
                $end = 1;
                last;
            }
            $length =~ m/\d+$/;
            $length = $&;
            push @contexts, $context;
            push @lengths, $length;
        }
        if ($end == 1)
        {
            last;
        }
        my $actualTime = findNext "execution time:.+ms\$";
        if (!defined($actualTime))
        {
            last;
        }
        $actualTime =~ s/execution time:\s*//g;
        $actualTime =~ s/\s*ms//g;
        $actualTime /= 1000;
        push @actualTimeData, $actualTime;

        my $contextStr = join " ", @contexts;
        push @contextData, $contextStr;
        my $lengthStr = join " ", @lengths;
        push @tokenData, $lengthStr;

        my $cmd = $cmdPrefix . "-c \"$contextStr\" -t \"$lengthStr\"";
        push @cmdPrefixData, $cmd;
        print STDERR "Calling $cmd\n";
        my $result = `$cmd`;

        my $projectedTime = "Not found";
        my @lines = split /\n/, $result;
        foreach my $tmpLine (@lines)
        {
            if ($tmpLine =~ m/total time:\s*[0-9\.]+/)
            {
                $projectedTime = $&;
                $projectedTime =~ s/total time:\s*//;
            }
        }
        print STDERR "  Projected iteration time: $projectedTime\n";
        push @projectedTimeData, $projectedTime;

        $iteration++;
    }
}
close FIN;

print STDOUT "Iteration,Efficiency,Projected,Actual,Context,Tokens,Command\n";
my $sumOfProjectedTime = 0;
my $sumOfActualTime = 0;
for (my $i = 0; $i < $iteration; $i++)
{
    my $efficiency = $projectedTimeData[$i] / $actualTimeData[$i];
    $sumOfProjectedTime += $projectedTimeData[$i];
    $sumOfActualTime += $actualTimeData[$i];
    print STDOUT "$i,$efficiency,$projectedTimeData[$i],$actualTimeData[$i],$contextData[$i],$tokenData[$i],$cmdPrefixData[$i]\n";
}
print STDOUT "==========================\n";
print STDOUT "Total,Efficiency,Projected,Actual\n";
my $overallEfficiency = $sumOfProjectedTime / $sumOfActualTime;
print STDOUT "$iteration,$overallEfficiency,$sumOfProjectedTime,$sumOfActualTime\n";

sub findNext
{
    my $regex = $_[0];
    #print "*************\nSearch for $regex\n";
    while (1)
    {
        my $line = <FIN>;
        if ($line eq "")
        {
            return undef;
        }
        chomp $line;
        #print "  Next line: $line\n";
        if ($line =~ m/$regex/)
        {
            #print "Matched\n********************\n";
            return $&;
        }
    }
}
