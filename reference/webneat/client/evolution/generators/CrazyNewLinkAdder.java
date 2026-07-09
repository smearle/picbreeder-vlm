/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.evolution.generators;

public class CrazyNewLinkAdder extends SequenceMutator {
	public CrazyNewLinkAdder() {
		final double colorProbability = client.ParameterTableInstance.get().getDouble("evolution", "add color link rate");
		final double greyProbability = client.ParameterTableInstance.get().getDouble("evolution", "add grey link rate");
		final double betweenProbability = client.ParameterTableInstance.get().getDouble("evolution", "add grey-color link rate");
		
		addMutator(new ProbabilisticMutator(new AddAcyclicLinkToAffinity("color"), colorProbability));
		addMutator(new ProbabilisticMutator(new AddAcyclicLinkToAffinity("grey"), greyProbability));
		addMutator(new ProbabilisticMutator(new AddAcyclicLinkBetweenAffinities("grey", "color"), betweenProbability));
	}
}
